import asyncio
from collections import OrderedDict
import math
from queue import Empty
import time
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from logging import Logger
from threading import Thread

import cv2
import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import BasePacket, FramePacket, DisparityDepthPacket
from depthai_sdk.classes.packet_handlers import QueuePacketHandler
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.pointcloud_component import PointcloudComponent
from depthai_sdk.components.stereo_component import StereoComponent
from numpy.typing import NDArray


DIMENSIONS_TO_MONO_RES = {
    (1280, 800): dai.MonoCameraProperties.SensorResolution.THE_800_P,
    (1280, 720): dai.MonoCameraProperties.SensorResolution.THE_720_P,
    (640, 400): dai.MonoCameraProperties.SensorResolution.THE_400_P,
}  # stereo camera component only accepts this subset of depthai_sdk.components.camera_helper.monoResolutions

DIMENSIONS_TO_COLOR_RES = {
    (4056, 3040): dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    (3840, 2160): dai.ColorCameraProperties.SensorResolution.THE_4_K,
    (1920, 1080): dai.ColorCameraProperties.SensorResolution.THE_1080_P,
}  # color camera component only accepts this subset of depthai_sdk.components.camera_helper.colorResolutions

MAX_GRPC_MESSAGE_BYTE_COUNT = 4194304  # Update this if the gRPC config ever changes


class WorkerManager(Thread):
    """
    WorkerManager checks the health of the OakCamera to make sure
    it is physically connected and functioning in a separate thread.
    """

    def __init__(
        self,
        oak: OakCamera,
        logger: Logger,
        reconfigure: Callable[[None], None],
    ) -> None:
        self.oak = oak
        self.logger = logger
        self.reconfigure = reconfigure

        self.running = True
        super().__init__()

    def run(self) -> None:
        self.logger.debug("Starting worker manager.")
        while self.running:
            self.logger.debug("Checking if worker must be reconfigured.")
            if self.oak.device.isClosed():
                self.logger.debug("Camera is closed. Reconfiguring worker.")
                self.reconfigure()
                self.running = False
            time.sleep(3)

    def stop(self) -> None:
        self.logger.debug("Stopping worker manager.")
        self.running = False


class MessageSynchronizer:
    """
    MessageSynchronizer manages synchronization of frame messages for color and depth data from OakCamera packet queues,
    maintaining an ordered dictionary of messages keyed chronologically by sequence number.
    """

    MAX_MSGS_SIZE = 50
    msgs: OrderedDict[int, Dict[str, BasePacket]]

    def __init__(self):
        # msgs maps frame sequence number to a dictionary that maps frame_type (i.e. "color" or "depth") to a data packet
        self.msgs = OrderedDict()

    def add_msg(
        self, msg: BasePacket, frame_type: Literal["color", "depth"], seq: int
    ) -> None:
        # Update recency if previously already stored in dict
        if seq in self.msgs:
            self.msgs.move_to_end(seq)

        self.msgs.setdefault(seq, {})[frame_type] = msg
        self._cleanup_msgs()

    def get_synced_msgs(self) -> Optional[Dict[str, BasePacket]]:
        for sync_msgs in self.msgs.values():
            if len(sync_msgs) == 2:  # has both color and depth
                return sync_msgs
        return None
    
    # def flush_msgs(self) -> None:
    #     self.msgs.clear()

    def get_most_recent_color_msg(self) -> Optional[BasePacket]:
        return self._get_most_recent_msg("color")

    def get_most_recent_depth_msg(self) -> Optional[BasePacket]:
        return self._get_most_recent_msg("depth")

    def _get_most_recent_msg(
        self, frame_type: Literal["color", "depth"]
    ) -> Optional[BasePacket]:
        # Traverse in reverse to get the most recent
        for msg_dict in reversed(self.msgs.values()):
            if frame_type in msg_dict:
                return msg_dict[frame_type]
        return None

    def _cleanup_msgs(self):
        while len(self.msgs) > self.MAX_MSGS_SIZE:
            self.msgs.popitem(last=False)  # remove oldest item


class CapturedData:
    """
    CapturedData is image data with the data as an np array,
    plus the timestamp it was captured at.
    """

    def __init__(self, np_array: NDArray, captured_at: float) -> None:
        self.np_array = np_array
        self.captured_at = captured_at


class Worker:
    """
    OakDModel <-> Worker <-> DepthAI SDK <-> DepthAI API (C++ core) <-> actual OAK-D
    """

    color_q_handler: Optional[QueuePacketHandler]
    depth_q_handler: Optional[QueuePacketHandler]
    pc_q_handler: Optional[QueuePacketHandler]

    def __init__(
        self,
        height: int,
        width: int,
        frame_rate: float,
        user_wants_color: bool,
        user_wants_depth: bool,
        user_wants_pc: bool,
        reconfigure: Callable[[None], None],
        logger: Logger,
    ) -> None:
        logger.info("Initializing worker.")

        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.user_wants_color = user_wants_color
        self.user_wants_depth = user_wants_depth
        self.user_wants_pc = user_wants_pc
        self.reconfigure = reconfigure
        self.logger = logger

        self.color_image = None
        self.depth_map = None
        self.pcd = None
        self.running = True

        self._init_oak_camera()
        self._config_oak_camera()
        self.oak.start()

        self.manager = WorkerManager(self.oak, logger, reconfigure)
        self.manager.start()

        self.message_synchronizer = MessageSynchronizer()

    async def get_synced_color_depth_data(self) -> Tuple[CapturedData, CapturedData]:
        while self.running:
            color_and_depth_output = self._try_get_synced_color_depth_data()
            if color_and_depth_output:
                return color_and_depth_output
            self.logger.debug("Waiting for synced color and depth frames...")
            await asyncio.sleep(0.001)

    def get_color_image(self) -> Optional[CapturedData]:
        color_msg: Optional[FramePacket] = (
            self.message_synchronizer.get_most_recent_color_msg()
        )
        if not color_msg:
            try:  # to get frame directly from queue
                color_q = self.color_q_handler.get_queue()
                color_msg = color_q.get(block=True, timeout=5)
            except Empty:
                raise Exception("Timed out waiting for color image data from queue.")

        color_output = self._process_color_frame(color_msg.frame)
        timestamp = color_msg.get_timestamp().total_seconds()
        return CapturedData(color_output, timestamp)

    def get_depth_map(self) -> Optional[CapturedData]:
        depth_msg: Optional[DisparityDepthPacket] = (
            self.message_synchronizer.get_most_recent_depth_msg()
        )
        if not depth_msg:
            try:  # to get frame directly from queue
                depth_q = self.depth_q_handler.get_queue()
                depth_msg = depth_q.get(block=True, timeout=5)
            except Empty:
                raise Exception("Timed out waiting for depth map data from queue.")

        depth_output = self._process_depth_frame(depth_msg.frame)
        timestamp = depth_msg.get_timestamp().total_seconds()
        return CapturedData(depth_output, timestamp)

    def get_pcd(self) -> CapturedData:
        pc_q = self.pc_q_handler.get_queue()
        try:
            pc_msg = pc_q.get(block=True, timeout=5)
            if pc_msg.points.nbytes > MAX_GRPC_MESSAGE_BYTE_COUNT:
                pc_output = self._downsample_pcd(pc_msg.points, pc_msg.points.nbytes)
            else:
                pc_output = pc_msg.points
            timestamp = pc_msg.get_timestamp().total_seconds()
            return CapturedData(pc_output, timestamp)
        except Empty:
            raise Exception("Timed out waiting for PCD.")

    def stop(self) -> None:
        """
        Handles stopping and closing of the camera worker.
        """
        self.logger.info("Stopping worker.")
        self.manager.stop()
        self.oak.close()

    def _try_get_synced_color_depth_data(
        self,
    ) -> Optional[Tuple[CapturedData, CapturedData]]:
        for frame_type, q_handler in [
            ("color", self.color_q_handler),
            ("depth", self.depth_q_handler),
        ]:
            queue_obj = q_handler.get_queue()
            with queue_obj.mutex:
                current_queue_msgs = queue_obj.queue

            for msg in current_queue_msgs:
                self.message_synchronizer.add_msg(
                    msg, frame_type, msg.get_sequence_num()
                )

        synced_color_msgs = self.message_synchronizer.get_synced_msgs()
        if synced_color_msgs:
            color_frame, depth_frame, timestamp = (
                synced_color_msgs["color"].frame,
                synced_color_msgs["depth"].frame,
                synced_color_msgs["color"].get_timestamp().total_seconds(),
            )
            color_data = CapturedData(self._process_color_frame(color_frame), timestamp)
            depth_data = CapturedData(self._process_depth_frame(depth_frame), timestamp)
            return color_data, depth_data

    def _init_oak_camera(self):
        """
        Blocks until the OakCamera is successfully initialized.
        """
        self.oak = None
        while not self.oak and self.running:
            try:
                self.oak = OakCamera()
            except Exception as e:
                self.logger.error(f"Error initializing OakCamera: {e}")
                time.sleep(1)

    def _config_oak_camera(self):
        """
        Safely configures the OakCamera.
        """
        try:
            stage = "color"
            color = self._configure_color()
            if color:
                self.color_q_handler = self.oak.queue(color, 30)

            stage = "stereo"
            stereo = self._configure_stereo(color)
            if stereo:
                self.depth_q_handler = self.oak.queue(stereo, 30)

            stage = "point cloud"
            pcc = self._configure_pc(stereo, color)
            if pcc:
                self.pc_q_handler = self.oak.queue(pcc, 5)

        except Exception as e:
            msg = f"Error configuring OakCamera at stage '{stage}': {e}"
            resolution_err_substr = "bigger than maximum at current sensor resolution"
            calibration_err_substr = "no Camera data available"
            if resolution_err_substr in str(e):
                msg += ". Please adjust 'height_px' and 'width_px' in your config to an accepted resolution."
            elif calibration_err_substr in str(e):
                msg += ". If using a non-integrated model, please check that the camera is calibrated properly."
            self.logger.error(msg)

    def _get_closest_resolution(
        self,
        dimensions_to_resolution: Mapping[
            Tuple[int, int],
            Union[
                dai.ColorCameraProperties.SensorResolution,
                dai.MonoCameraProperties.SensorResolution,
            ],
        ],
    ) -> Union[
        dai.ColorCameraProperties.SensorResolution,
        dai.MonoCameraProperties.SensorResolution,
    ]:
        """
        Intakes a dict mapping width/height to a resolution and calculates the closest
        supported resolution to the width and height from the config.

        Args:
            resolutions (dict)
        Returns:
            Union[dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution]
        """

        def euclidian_distance(width_and_height: Tuple[int, int]) -> float:
            w1, h1 = width_and_height
            w2, h2 = self.width, self.height
            return math.sqrt((w1 - w2) ** 2 + (h1 - h2) ** 2)

        closest = min(
            dimensions_to_resolution.keys(),
            key=euclidian_distance,
        )
        return dimensions_to_resolution[closest]

    def _configure_color(self) -> Optional[CameraComponent]:
        """
        Creates and configures color component— or doesn't
        (based on the config).

        Returns:
            Union[CameraComponent, None]
        """
        if self.user_wants_color:
            self.logger.debug("Creating color camera component.")
            resolution = self._get_closest_resolution(DIMENSIONS_TO_COLOR_RES)
            self.logger.debug(
                f"Closest color resolution to inputted height width is: {resolution}"
            )
            color = self.oak.camera("color", fps=self.frame_rate, resolution=resolution)
            color.node.setPreviewSize(self.width, self.height)
            color.node.setVideoSize(self.width, self.height)
            return color
        return None

    def _configure_stereo(
        self, color: Optional[CameraComponent]
    ) -> Optional[StereoComponent]:
        """
        Creates and configures stereo depth component— or doesn't
        (based on the config).

        Returns:
            Union[StereoComponent, None]
        """
        if self.user_wants_depth:
            self.logger.debug("Creating stereo depth component.")
            # TODO RSDK-5633: Figure out how to use DepthAI to adjust depth output size.
            # Right now it's being handled as a manual crop resize in _set_depth_map.
            # The below commented out code should fix this
            # ... but DepthAI hasn't implemented config_camera for mono cameras yet.
            # mono_left = oak.camera(dai.CameraBoardSocket.LEFT, fps=self.frame_rate)
            # mono_left.config_camera((self.width, self.height))
            # mono_right = oak.camera(dai.CameraBoardSocket.RIGHT, fps=self.frame_rate)
            # mono_right.config_camera((self.width, self.height))
            # # then pass mono_left and mono_right to the oak.stereo initialization
            resolution = self._get_closest_resolution(DIMENSIONS_TO_MONO_RES)
            self.logger.debug(
                f"Closest mono resolution to inputted height width is: {resolution}"
            )
            cam_left = self.oak.camera(
                dai.CameraBoardSocket.CAM_B,  # Same as CameraBoardSocket.LEFT
                resolution,
                self.frame_rate,
            )
            cam_right = self.oak.camera(
                dai.CameraBoardSocket.CAM_C,  # Same as CameraBoardSocket.RIGHT
                resolution,
                self.frame_rate,
            )
            stereo = self.oak.stereo(resolution, self.frame_rate, cam_left, cam_right)
            if color:
                # Ensures camera alignment between color and depth
                stereo.config_stereo(align=color)
            return stereo
        return None

    def _configure_pc(
        self, stereo: Optional[StereoComponent], color: Optional[CameraComponent]
    ) -> Optional[PointcloudComponent]:
        """
        Creates and configures point cloud component— or doesn't
        (based on the config)

        Returns:
            Union[PointcloudComponent, None]
        """
        if self.user_wants_pc:
            self.logger.debug("Creating point cloud component.")
            pcc = self.oak.create_pointcloud(stereo, color)
            return pcc
        return None

    def _process_color_frame(self, arr: NDArray) -> NDArray:
        # DepthAI outputs BGR; convert to RGB
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def _process_depth_frame(self, arr: NDArray) -> NDArray:
        if arr.shape[0] > self.height and arr.shape[1] > self.width:
            self.logger.debug(
                f"Outputted depth map's shape is greater than specified in config: {arr.shape}; Manually resizing to {(self.height, self.width)}."
            )
            top_left_x = (arr.shape[1] - self.width) // 2
            top_left_y = (arr.shape[0] - self.height) // 2
            return arr[
                top_left_y : top_left_y + self.height,
                top_left_x : top_left_x + self.width,
            ]
        return arr

    def _downsample_pcd(self, arr: NDArray, byte_count: int) -> NDArray:
        factor = byte_count // MAX_GRPC_MESSAGE_BYTE_COUNT + 1
        self.logger.warn(
            f"PCD bytes ({byte_count}) > max gRPC bytes count ({MAX_GRPC_MESSAGE_BYTE_COUNT}). Subsampling by 1/{factor}."
        )
        arr = arr[::factor, ::factor, :]
        return arr
