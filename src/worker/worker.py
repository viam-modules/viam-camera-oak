import asyncio
from collections import OrderedDict
import math
from queue import Empty
from threading import Lock
import time
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from logging import Logger

import cv2
import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.classes.packet_handlers import BasePacket, QueuePacketHandler
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.stereo_component import StereoComponent
from numpy.typing import NDArray

from src.helpers.shared import CapturedData, Sensor
from src.helpers.config import OakConfig


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


def get_closest_dai_resolution(
    width,
    height,
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
        w2, h2 = width, height
        return math.sqrt((w1 - w2) ** 2 + (h1 - h2) ** 2)

    closest = min(
        dimensions_to_resolution.keys(),
        key=euclidian_distance,
    )
    return dimensions_to_resolution[closest]


MAX_GRPC_MESSAGE_BYTE_COUNT = 4194304  # Update this if the gRPC config ever changes


class SensorAndQueueHandler:
    def __init__(self, sensor: Sensor, queue_handler: QueuePacketHandler):
        self.sensor = sensor
        self.queue_handler = queue_handler


class Worker:
    """
    oak.py <-> worker.py <-> DepthAI SDK <-> DepthAI API (C++ core) <-> actual camera
    """

    oak: Optional[OakCamera]
    color_handlers: Optional[List[SensorAndQueueHandler]]
    stereo_queue_handler: Optional[QueuePacketHandler]
    pc_queue_handler: Optional[QueuePacketHandler]

    should_exec: bool
    configured: bool
    running: bool

    def __init__(
        self,
        oak_config: OakConfig,
        user_wants_pc: bool,
        # reconfigure: Callable[[None], None],
        logger: Logger,
    ) -> None:
        logger.info("Initializing worker.")

        self.cfg = oak_config
        self.user_wants_pc = user_wants_pc
        # self.reconfigure = reconfigure
        self.logger = logger

        self.oak = None
        self.color_handlers = None
        self.stereo_queue_handler = None
        self.pc_queue_handler = None

        # Flag for stopping execution and busy loops
        self.should_exec = True
        # Execution status
        self.configured = False
        self.running = False

    def configure(self):
        self._init_oak_camera()

        if (
            not self.should_exec
        ):  # possible SIGTERM between initializing OakCamera and here
            self.logger.debug("Stopping configuration due to termination signal")
            return

        self._config_oak_camera()

        self.message_synchronizer = MessageSynchronizer()
        self.configured = True

    def start(self):
        if self.oak:
            self.oak.start()
            self.logger.info("Started OakCamera")
        else:
            raise AttributeError(
                "oak.start() called before oak was assigned. Must configure worker first."
            )
        self.running = True

    async def get_synced_color_depth_data(self) -> Tuple[CapturedData, CapturedData]:
        if not self.running:
            raise Exception("Error getting camera output: Worker is not currently running.")
        if len(self.color_handlers) != 1 or not self.stereo_queue_handler:
            raise Exception("Precondition error: must have exactly 1 color sensor and 2 mono sensors for synced data")

        while self.should_exec:
            self.message_synchronizer._add_msgs_from_queue(
                "color", self.color_handlers[0].queue_handler
            )
            self.message_synchronizer._add_msgs_from_queue(
                "depth", self.stereo_queue_handler
            )

            synced_msgs = self.message_synchronizer.get_synced_msgs()
            if synced_msgs:
                color_frame, depth_frame, timestamp = (
                    synced_msgs["color"].frame,
                    synced_msgs["depth"].frame,
                    synced_msgs["color"].get_timestamp().total_seconds(),
                )
                if self.cfg.sensors.primary_sensor.color_order == "rgb":
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_data = CapturedData(
                    color_frame, timestamp
                )
                depth_data = CapturedData(
                    self._process_depth_frame(self.cfg.sensors.stereo_pair[0], depth_frame), timestamp
                )
                return color_data, depth_data

            self.logger.debug("Waiting for synced color and depth frames...")
            await asyncio.sleep(0.001)

    def get_color_output(self, requested_sensor: Sensor):
        if not self.running:
            raise Exception("Error getting color frame: Worker is not currently running.")
        if requested_sensor.sensor_type != "color":
            raise Exception("Error getting color frame: requested sensor is not a color sensor")

        for sensor_and_queue in self.color_handlers:
            sensor = sensor_and_queue.sensor
            q = sensor_and_queue.queue_handler
            if requested_sensor.socket != sensor.socket:
                continue
            try:
                msg = q.get_queue().get(block=True, timeout=5)
            except Empty:
                raise Empty("Error getting color frame: frame queue is empty.")
            color_output = msg.frame
            if sensor.color_order == "rgb":
                color_output = cv2.cvtColor(msg.frame, cv2.COLOR_BGR2RGB)
            timestamp = msg.get_timestamp().total_seconds()
            return CapturedData(color_output, timestamp)
        
        raise Exception("Error getting color frame: requested sensor not registered in worker")

    def get_depth_output(self):
        if not self.running:
            raise Exception("Error getting depth frame: Worker is not currently running.")
        if self.stereo_queue_handler is None:
            raise Exception("Error getting depth frame: stereo depth component is not configured")

        try:
            msg = self.stereo_queue_handler.get_queue().get(block=True, timeout=5)
        except Empty:
            raise Empty("Error getting depth frame: frame queue is empty.")
        depth_output = self._process_depth_frame(self.cfg.sensors.stereo_pair[0], msg.frame)
        timestamp = msg.get_timestamp().total_seconds()
        return CapturedData(depth_output, timestamp)

    def get_pcd(self) -> CapturedData:
        if not self.running:
            raise Exception("Error getting PCD: Worker is not currently running.")
        try:
            pc_msg = self.pc_queue_handler.get_queue().get(block=True, timeout=5)
            if pc_msg.points.nbytes > MAX_GRPC_MESSAGE_BYTE_COUNT:
                pc_output = self._downsample_pcd(pc_msg.points, pc_msg.points.nbytes)
            else:
                pc_output = pc_msg.points
            timestamp = pc_msg.get_timestamp().total_seconds()
            return CapturedData(pc_output, timestamp)
        except Empty:
            raise Exception("Error getting PCD: timed out waiting for PCD from queue.")

    def stop(self) -> None:
        """
        Handles closing resources and exiting logic in worker.
        """
        self.logger.debug("Stopping worker.")
        self.should_exec = False
        self.configured = False
        self.running = False
        if self.oak and self.oak.device and not self.oak.device.isClosed():
            self.oak.close()

    def reset(self) -> None:
        self.logger.debug("Resetting worker.")
        self.should_exec = True
        self.configured = False
        self.running = False
        if self.oak and self.oak.device and not self.oak.device.isClosed():
            self.oak.close()
            self.oak = None

    def _init_oak_camera(self):
        """
        Blocks until the OakCamera is successfully initialized.
        """
        self.oak = None
        while not self.oak and self.should_exec:
            try:
                self.oak = OakCamera()
                self.logger.info("Successfully initialized OakCamera.")
            except Exception as e:
                self.logger.error(f"Error initializing OakCamera: {e}")
                time.sleep(1)

    def _config_oak_camera(self):
        """
        Safely configures the OakCamera.
        """
        try:
            stage = "color"
            color_component = self._configure_color()

            stage = "stereo"
            stereo_component = self._configure_stereo(color_component)

            stage = "point cloud"
            self._configure_pc(color_component, stereo_component)

            self.logger.info("Successfully configured OakCamera")
        except Exception as e:
            msg = f"Error configuring OakCamera at stage '{stage}': {e}"
            resolution_err_substr = "bigger than maximum at current sensor resolution"
            calibration_err_substr = "no Camera data available"
            if resolution_err_substr in str(e):
                msg += ". Please adjust 'height_px' and 'width_px' in your config to an accepted resolution."
            elif calibration_err_substr in str(e):
                msg += ". If using a non-integrated model, please check that the camera is calibrated properly."
            self.logger.error(msg)

    def _configure_color(self) -> Optional[CameraComponent]:
        """
        Creates and configures color components— or doesn't
        (based on the config).
        """
        primary_color_component = None
        color_handlers = []
        for sensor in self.cfg.sensors.color_sensors:
            if sensor.sensor_type != "color":
                continue
            self.logger.debug("Creating color camera component.")
            resolution = get_closest_dai_resolution(
                sensor.width, sensor.height, DIMENSIONS_TO_COLOR_RES
            )
            self.logger.debug(
                f"Closest color resolution to inputted height & width is: {resolution}"
            )
            color = self.oak.camera(
                f"{sensor.socket},c", fps=sensor.frame_rate, resolution=resolution
            )
            if not isinstance(color.node, dai.node.ColorCamera):
                raise Exception(
                    f"Underlying DepthAI or hardware configuration error: expected node to be ColorCamera, got {type(color)} for socket {sensor.socket}"
                )
            if primary_color_component is None:
                primary_color_component = color
            color.node.setPreviewSize(sensor.width, sensor.height)
            color.node.setVideoSize(sensor.width, sensor.height)
            color.node.setInterleaved(sensor.interleaved)
            if sensor.color_order == "bgr":
                color.node.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            elif sensor.color_order == "rgb":
                color.node.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            color_handlers.append(SensorAndQueueHandler(sensor, self.oak.queue(color, 30)))
        self.color_handlers = color_handlers
        return primary_color_component  # stereo and pc will be aligned to the primary color sensor if requested & possible

    def _configure_stereo(self, color_component: CameraComponent) -> Optional[StereoComponent]:
        """
        Creates and configures stereo depth component— or doesn't
        (based on the config).
        """
        def make_mono_cam(mono: Sensor) -> Tuple[CameraComponent, dai.MonoCameraProperties.SensorResolution]:
            resolution = get_closest_dai_resolution(mono.width, mono.height, DIMENSIONS_TO_MONO_RES)
            self.logger.debug(
                f"Closest mono resolution: {resolution}. Inputted width & height: ({mono.width}, {mono.height})"
            )
            mono_component = self.oak.camera(
                mono.socket,
                resolution,
                mono.frame_rate,
            )
            return mono_component, resolution

        stereo_pair = self.cfg.sensors.stereo_pair
        if stereo_pair:
            self.logger.debug("Creating stereo depth component.")
            mono1, mono2 = stereo_pair
            mono1_component, mono1_res = make_mono_cam(mono1)
            mono2_component, mono2_res = make_mono_cam(mono2)
            if mono1_res != mono2_res:
                self.logger.warn(f"Stereo depth pair mono cams configured with different resolutions. Defaulting to {mono1_res}")
            stereo = self.oak.stereo(mono1_res, mono1.frame_rate, mono1_component, mono2_component)
            if color_component:
                # Ensures camera alignment between color and depth
                stereo.config_stereo(align=color_component)
            self.stereo_queue_handler = self.oak.queue(stereo, max_size=30)
            return stereo

    def _configure_pc(self, color_component: CameraComponent, stereo_component: StereoComponent):
        """
        Creates and configures point cloud component— or doesn't
        (based on the config)
        """
        if stereo_component:
            self.logger.debug("Creating point cloud component.")
            pc_component = self.oak.create_pointcloud(stereo_component, color_component)
            self.pc_queue_handler = self.oak.queue(pc_component, 5)

    def _process_depth_frame(self, sensor: Sensor, arr: NDArray) -> NDArray:
        if arr.shape[0] > sensor.height and arr.shape[1] > sensor.width:
            self.logger.debug(
                f"Outputted depth map's shape is greater than specified in config: {arr.shape}; Manually resizing to {(sensor.height, sensor.width)}."
            )
            top_left_x = (arr.shape[1] - sensor.width) // 2
            top_left_y = (arr.shape[0] - sensor.height) // 2
            return arr[
                top_left_y : top_left_y + sensor.height,
                top_left_x : top_left_x + sensor.width,
            ]
        return arr

    def _downsample_pcd(self, arr: NDArray, byte_count: int) -> NDArray:
        factor = byte_count // MAX_GRPC_MESSAGE_BYTE_COUNT + 1
        self.logger.warn(
            f"PCD bytes ({byte_count}) > max gRPC bytes count ({MAX_GRPC_MESSAGE_BYTE_COUNT}). Subsampling by 1/{factor}."
        )
        arr = arr[::factor, ::factor, :]
        return arr


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
