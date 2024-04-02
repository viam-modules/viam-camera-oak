import asyncio
from collections import OrderedDict
import math
from queue import Empty
import time
from typing import (
    Callable,
    Dict,
    Literal,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from logging import Logger
from threading import Lock

import cv2
import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import BasePacket
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


class MessageSynchronizer:
    """
    MessageSynchronizer manages synchronization of frame messages for color and depth data from OakCamera packet queues,
    maintaining an ordered dictionary of messages keyed chronologically by sequence number.
    """

    MAX_MSGS_SIZE = 50
    msgs: OrderedDict[int, Dict[str, BasePacket]]
    write_lock: Lock

    def __init__(self):
        # msgs maps frame sequence number to a dictionary that maps frame_type (i.e. "color" or "depth") to a data packet
        self.msgs = OrderedDict()
        self.write_lock = Lock()

    def add_msg(
        self, msg: BasePacket, frame_type: Literal["color", "depth"], seq: int
    ) -> None:
        with self.write_lock:
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

    def _add_msgs_from_queue(
        self, frame_type: Literal["color", "depth"], queue_handler: QueuePacketHandler
    ) -> None:
        queue_obj = queue_handler.get_queue()
        with queue_obj.mutex:
            q_snapshot = list(queue_obj.queue)

        for msg in q_snapshot:
            self.add_msg(msg, frame_type, msg.get_sequence_num())

    def get_most_recent_msg(
        self, q_handler: QueuePacketHandler, frame_type: Literal["color", "depth"]
    ) -> Optional[BasePacket]:
        self._add_msgs_from_queue(frame_type, q_handler)
        while len(self.msgs) < 1:
            self._add_msgs_from_queue(frame_type, q_handler)
            time.sleep(0.1)
        # Traverse in reverse to get the most recent
        for msg_dict in reversed(self.msgs.values()):
            if frame_type in msg_dict:
                return msg_dict[frame_type]
        raise Exception(f"No message of type '{frame_type}' in frame queue.")

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
    oak.py <-> worker.py <-> DepthAI SDK <-> DepthAI API (C++ core) <-> actual camera
    """

    oak: Optional[OakCamera]
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

        # Args -> states
        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.user_wants_color = user_wants_color
        self.user_wants_depth = user_wants_depth
        self.user_wants_pc = user_wants_pc
        self.reconfigure = reconfigure
        self.logger = logger

        # Managed objects
        self.oak = None
        self.color_q_handler = None
        self.depth_q_handler = None
        self.pc_q_handler = None

        # Flags for stopping busy loops
        self.running = False
        self.starting_up = False

    def start(self):
        self.starting_up = True
        self._init_oak_camera()

        if (
            not self.starting_up
        ):  # possible SIGTERM between initializing OakCamera and here
            return

        self._config_oak_camera()
        self.oak.start()

        self.message_synchronizer = MessageSynchronizer()
        self.running = True
        self.starting_up = False

    async def get_synced_color_depth_data(self) -> Tuple[CapturedData, CapturedData]:
        while self.running:
            self.message_synchronizer._add_msgs_from_queue(
                "color", self.color_q_handler
            )
            self.message_synchronizer._add_msgs_from_queue(
                "depth", self.depth_q_handler
            )

            color_and_depth_data = self._capture_synced_color_depth_data()
            if color_and_depth_data:
                return color_and_depth_data

            self.logger.debug("Waiting for synced color and depth frames...")
            await asyncio.sleep(0.001)

    def get_color_image(self) -> Optional[CapturedData]:
        color_msg = self.message_synchronizer.get_most_recent_msg(
            self.color_q_handler, "color"
        )
        color_output = self._process_color_frame(color_msg.frame)
        timestamp = color_msg.get_timestamp().total_seconds()
        return CapturedData(color_output, timestamp)

    def get_depth_map(self) -> Optional[CapturedData]:
        depth_msg = self.message_synchronizer.get_most_recent_msg(
            self.depth_q_handler, "depth"
        )
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
        self.starting_up = False
        self.running = False
        if self.oak:
            self.oak.close()

    def _capture_synced_color_depth_data(
        self,
    ) -> Optional[Tuple[CapturedData, CapturedData]]:
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
        while not self.oak and self.starting_up:
            try:
                self.oak = OakCamera()
                self.logger.debug("Successfully initialized OakCamera.")
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
        dimensions_to_resolution: Dict[
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
