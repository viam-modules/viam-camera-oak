import asyncio
from collections import OrderedDict
import math
from queue import Empty
from threading import Lock
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from viam.logging import getLogger

import cv2
import depthai as dai
from numpy.typing import NDArray
import numpy as np

from src.components.helpers.shared import CapturedData, Sensor
from src.components.helpers.config import OakConfig

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

LOGGER = getLogger("viam-oak-worker-logger")


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


class SensorAndQueue:
    def __init__(self, sensor: Sensor, q: dai.DataOutputQueue):
        self.sensor = sensor
        self.queue = q


class Worker:
    """
    oak.py <-> worker.py <-> DepthAI SDK <-> DepthAI API (C++ core) <-> actual camera
    """

    pipeline: Optional[dai.Pipeline]
    device: Optional[dai.Device]
    color_sensor_queues: Optional[List[SensorAndQueue]]
    depth_queue: Optional[dai.DataOutputQueue]
    pcd_queue: Optional[dai.DataOutputQueue]

    should_exec: bool
    configured: bool
    running: bool

    depth_stream_name = "depth"
    pc_stream_name = "pc"

    def __init__(
        self,
        oak_config: OakConfig,
        user_wants_pc: bool,
    ) -> None:
        LOGGER.info("Initializing worker.")
        self.cfg = oak_config
        self.user_wants_pc = user_wants_pc

        self.device = None
        self.pipeline = None
        self.color_sensor_queues = None
        self.depth_queue = None
        self.pcd_queue = None

        # Flag for stopping execution and busy loops
        self.should_exec = True
        # Execution status
        self.configured = False
        self.running = False

    def configure(self):
        def configure_color() -> Optional[dai.node.ColorCamera]:
            """
            Creates and configures color cameras or doesn't
            (based on the config).
            """
            primary_color_cam = None
            for sensor in self.cfg.sensors.color_sensors:
                if sensor.sensor_type != "color":
                    continue
                LOGGER.debug("Creating color camera component.")
                resolution = get_closest_dai_resolution(
                    sensor.width, sensor.height, DIMENSIONS_TO_COLOR_RES
                )
                LOGGER.debug(
                    f"Closest color resolution to inputted height & width is: {resolution}"
                )

                # Define source and output
                color_cam = self.pipeline.create(dai.node.ColorCamera)
                xout_color = self.pipeline.create(dai.node.XLinkOut)

                xout_color.setStreamName(sensor.get_unique_name())

                # Properties
                color_cam.setPreviewSize(sensor.width, sensor.height)
                color_cam.setInterleaved(sensor.interleaved)
                color_cam.setBoardSocket(sensor.socket)
                color_cam.setFps(sensor.frame_rate)
                if sensor.color_order == "bgr":
                    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
                elif sensor.color_order == "rgb":
                    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

                # Linking
                color_cam.preview.link(xout_color.input)

                if primary_color_cam is None:
                    primary_color_cam = color_cam
            return primary_color_cam

        def configure_stereo() -> Optional[dai.node.StereoDepth]:
            """
            Creates and configures the pipeline for stereo depth— or doesn't
            (based on the config).
            """

            def make_mono_cam(sensor: Sensor) -> dai.node.MonoCamera:
                resolution = get_closest_dai_resolution(
                    sensor.width, sensor.height, DIMENSIONS_TO_MONO_RES
                )
                LOGGER.debug(
                    f"Closest mono resolution: {resolution}. Inputted width & height: ({sensor.width}, {sensor.height})"
                )

                mono_cam = self.pipeline.create(dai.node.MonoCamera)
                mono_cam.setResolution(resolution)
                mono_cam.setBoardSocket(sensor.socket)
                mono_cam.setFps(sensor.frame_rate)
                return mono_cam

            stereo_pair = self.cfg.sensors.stereo_pair
            depth = None
            if stereo_pair:
                LOGGER.debug("Creating stereo depth component.")
                mono1, mono2 = stereo_pair
                mono_cam_1 = make_mono_cam(mono1)
                mono_cam_2 = make_mono_cam(mono2)

                depth = self.pipeline.create(dai.node.StereoDepth)
                depth.setDefaultProfilePreset(
                    dai.node.StereoDepth.PresetMode.HIGH_DENSITY
                )
                depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

                mono_cam_1.out.link(depth.left)
                mono_cam_2.out.link(depth.right)

                xout_depth = self.pipeline.create(dai.node.XLinkOut)
                xout_depth.setStreamName(self.depth_stream_name)
                depth.disparity.link(xout_depth.input)

                color_sensors = self.cfg.sensors.color_sensors
                if color_sensors and len(color_sensors) > 0:
                    sensor_color_align = color_sensors[0]
                    depth.setDepthAlign(sensor_color_align.socket)
            return depth

        def configure_pc(
            color_node: Optional[dai.node.ColorCamera],
            depth_node: Optional[dai.node.StereoDepth],
        ) -> None:
            """
            Creates and configures point clouds— or doesn't
            (based on the config)
            """
            if depth_node and self.user_wants_pc:
                pointcloud = self.pipeline.create(dai.node.PointCloud)
                sync = self.pipeline.create(dai.node.Sync)
                xOut = self.pipeline.create(dai.node.XLinkOut)

                xOut.setStreamName(self.pc_stream_name)
                xOut.input.setBlocking(False)

                # Link the nodes
                depth_node.depth.link(pointcloud.inputDepth)
                if color_node:
                    color_node.isp.link(sync.inputs["rgb"])
                pointcloud.outputPointCloud.link(sync.inputs["pcl"])
                sync.out.link(xOut.input)

        self.pipeline = dai.Pipeline()
        try:
            stage = "color"
            color_node = configure_color()

            stage = "stereo"
            depth_node = configure_stereo()

            stage = "point cloud"
            configure_pc(color_node, depth_node)

            LOGGER.info("Successfully configured pipeline.")
        except Exception as e:
            msg = f"Error configuring pipeline at stage '{stage}': {e}"
            resolution_err_substr = "bigger than maximum at current sensor resolution"
            calibration_err_substr = "no Camera data available"
            if resolution_err_substr in str(e):
                msg += ". Please adjust 'height_px' and 'width_px' in your config to an accepted resolution."
            elif calibration_err_substr in str(e):
                msg += ". If using a non-integrated model, please check that the camera is calibrated properly."
            LOGGER.error(msg)

        self.message_synchronizer = MessageSynchronizer()
        self.configured = True

    async def start(self):
        self.device = None
        while not self.device and self.should_exec:
            try:
                self.device = dai.Device(self.pipeline)
                LOGGER.debug("Successfully initialized device.")
            except Exception as e:
                LOGGER.error(f"Error initializing device: {e}")
                await asyncio.sleep(1)

        self.color_sensor_queues: List[SensorAndQueue] = []
        for cs in self.cfg.sensors.color_sensors:
            q = self.device.getOutputQueue(cs.get_unique_name(), 30, blocking=False)
            self.color_sensor_queues.append(SensorAndQueue(cs, q))

        if self.cfg.sensors.stereo_pair:
            self.depth_queue = self.device.getOutputQueue(
                self.depth_stream_name, 30, blocking=False
            )
            if self.user_wants_pc:
                self.pcd_queue = self.device.getOutputQueue(
                    self.pc_stream_name, 5, blocking=False
                )

        should_get_synced_color_depth_outputs = len(
            self.color_sensor_queues
        ) == 1 and bool(self.depth_queue)
        if should_get_synced_color_depth_outputs:
            self.color_sensor_queues[0].queue.addCallback(
                self.message_synchronizer.add_color_msg
            )
            self.depth_queue.addCallback(self.message_synchronizer.add_depth_msg)
            self.message_synchronizer.callbacks_set = True

        self.running = True
        LOGGER.info("Successfully started camera worker.")

    async def get_synced_color_depth_data(self) -> Tuple[CapturedData, CapturedData]:
        if not self.running:
            raise Exception(
                "Error getting camera output: Worker is not currently running."
            )
        if len(self.color_sensor_queues) != 1 or not self.depth_queue:
            raise Exception(
                "Precondition error: must have exactly 1 color sensor and 2 mono sensors for synced data"
            )

        while self.should_exec:
            synced_msgs = self.message_synchronizer.get_synced_msgs()
            if synced_msgs:
                color_frame, depth_frame, timestamp = (
                    synced_msgs["color"].getCvFrame(),
                    synced_msgs["depth"].getCvFrame(),
                    synced_msgs["color"].getTimestamp().total_seconds(),
                )
                if self.cfg.sensors.primary_sensor.color_order == "rgb":
                    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_data = CapturedData(color_frame, timestamp)
                depth_data = CapturedData(
                    self._process_depth_frame(
                        self.cfg.sensors.stereo_pair[0], depth_frame
                    ),
                    timestamp,
                )
                return color_data, depth_data

            LOGGER.debug("Waiting for synced color and depth frames...")
            await asyncio.sleep(0.001)

    def get_color_output(self, requested_sensor: Sensor):
        if not self.running:
            raise Exception(
                "Error getting color frame: Worker is not currently running."
            )
        if requested_sensor.sensor_type != "color":
            raise Exception(
                "Error getting color frame: requested sensor is not a color sensor"
            )

        for sensor_and_queue in self.color_sensor_queues:
            this_sensor = sensor_and_queue.sensor
            if requested_sensor.socket_str != this_sensor.socket_str:
                continue

            q = sensor_and_queue.queue
            try:
                msg = q.get()
                frame = msg.getCvFrame()
            except Exception:
                raise Empty("Error getting color frame: frame queue is empty.")
            if this_sensor.color_order == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = msg.getTimestamp().total_seconds()
            return CapturedData(frame, timestamp)

        raise Exception(
            "Error getting color frame: requested sensor not registered in worker"
        )

    def get_depth_output(self):
        if not self.running:
            raise Exception(
                "Error getting depth frame: Worker is not currently running."
            )
        if not self.depth_queue:
            raise Exception(
                "Error getting depth frame: stereo depth frame queue is not configured"
            )

        try:
            msg = self.depth_queue.get()
        except Empty:
            raise Empty("Error getting depth frame: frame queue is empty.")
        depth_output = self._process_depth_frame(
            self.cfg.sensors.stereo_pair[0], msg.getCvFrame()
        )
        timestamp = msg.getTimestamp().total_seconds()
        return CapturedData(depth_output, timestamp)

    def get_pcd(self) -> CapturedData:
        if not self.running:
            raise Exception("Error getting PCD: Worker is not currently running.")
        if not self.user_wants_pc:
            raise Exception(
                "Critical dev logic error getting PCD: get_pcd() called without toggling user_wants_pc."
            )
        if not self.depth_queue:
            raise Exception(
                "Error getting PCD: depth frame queue not configured for current OAK camera."
            )
        try:
            pc_msg = self.pcd_queue.get()
            pc_obj = pc_msg["pcl"]
            points = pc_obj.getPoints().astype(np.float64)
            if points.nbytes > MAX_GRPC_MESSAGE_BYTE_COUNT:
                pc_output = self._downsample_pcd(points, points.nbytes)
            else:
                pc_output = points
            timestamp = pc_msg.getTimestamp().total_seconds()
            return CapturedData(pc_output, timestamp)
        except Empty:
            raise Exception("Error getting PCD: timed out waiting for PCD from queue.")

    def stop(self) -> None:
        """
        Handles closing resources and exiting logic in worker.
        """
        LOGGER.debug("Stopping worker.")
        self.should_exec = False
        self.configured = False
        self.running = False
        if self.device:
            self.device.close()
        if self.pipeline:
            self.pipeline = None

    def reset(self) -> None:
        LOGGER.debug("Resetting worker.")
        self.stop()
        self.should_exec = True

    def _process_depth_frame(self, sensor: Sensor, arr: NDArray) -> NDArray:
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)

        if arr.shape[0] > sensor.height and arr.shape[1] > sensor.width:
            LOGGER.debug(
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
        LOGGER.warn(
            f"PCD bytes ({byte_count}) > max gRPC bytes count ({MAX_GRPC_MESSAGE_BYTE_COUNT}). Subsampling by 1/{factor}."
        )
        if arr.ndim == 2:
            arr = arr[::factor, :]
        else:
            raise ValueError(f"Unexpected point cloud array dimensions: {arr.ndim}")
        return arr


class MessageSynchronizer:
    """
    MessageSynchronizer manages synchronization of frame messages for color and depth data packet queues,
    maintaining an ordered dictionary of messages keyed chronologically by sequence number.
    """

    MAX_MSGS_SIZE = 50
    msgs: OrderedDict[int, Dict[str, dai.ADatatype]]
    write_lock: Lock
    callbacks_set: bool

    def __init__(self):
        # msgs maps frame sequence number to a dictionary that maps frame_type (i.e. "color" or "depth") to a data packet
        self.msgs = OrderedDict()
        self.write_lock = Lock()
        self.callbacks_set = False

    def get_synced_msgs(self) -> Optional[Dict[str, dai.ADatatype]]:
        if not self.callbacks_set:
            raise Exception(
                "Data queue callbacks were not set. Cannot get synced messages."
            )
        for sync_msgs in self.msgs.values():
            if len(sync_msgs) == 2:  # has both color and depth
                return sync_msgs
        return None

    def add_color_msg(self, msg: dai.ADatatype) -> None:
        self._add_msg(msg, "color", msg.getSequenceNum())

    def add_depth_msg(self, msg: dai.ADatatype) -> None:
        self._add_msg(msg, "depth", msg.getSequenceNum())

    def _add_msg(
        self, msg: dai.ADatatype, frame_type: Literal["color", "depth"], seq: int
    ) -> None:
        with self.write_lock:
            # Update recency if previously already stored in dict
            if seq in self.msgs:
                self.msgs.move_to_end(seq)

            self.msgs.setdefault(seq, {})[frame_type] = msg
            self._cleanup_msgs()

    def _cleanup_msgs(self):
        while len(self.msgs) > self.MAX_MSGS_SIZE:
            self.msgs.popitem(last=False)  # remove oldest item
