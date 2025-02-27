import asyncio
from collections import OrderedDict
from logging import Logger
import math
from queue import Empty
from threading import Lock
from typing import (
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

from viam.logging import getLogger
from viam.errors import ViamError

import cv2
import depthai as dai
from numpy.typing import NDArray
import numpy as np

from src.components.helpers.shared import CapturedData
from src.config import OakConfig, OakDConfig, YDNConfig, Sensor

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

MAX_GRPC_MESSAGE_BYTE_COUNT = 33554432  # 32MB gRPC message size limit
MAX_COLOR_DEPTH_QUEUE_SIZE = 5
MAX_PC_QUEUE_SIZE = 5
MAX_YDN_QUEUE_SIZE = 5
MAX_MSG_SYCHRONIZER_MSGS_SIZE = 50


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
        dimensions_to_resolution (dict)
    Returns:
        Union[dai.ColorCameraProperties.SensorResolution, dai.MonoCameraProperties.SensorResolution]
    """

    def euclidean_distance(width_and_height: Tuple[int, int]) -> float:
        w1, h1 = width_and_height
        w2, h2 = width, height
        return math.sqrt((w1 - w2) ** 2 + (h1 - h2) ** 2)

    # Filter for only dimensions that are larger or equal to the required width and height
    valid_dimensions = {
        dim: res
        for dim, res in dimensions_to_resolution.items()
        if dim[0] >= width and dim[1] >= height
    }

    if not valid_dimensions:
        raise ViamError(
            f"Received width x height: {width} x {height}, but no valid resolutions are larger or equal to the requested dimensions."
        )

    closest = min(
        valid_dimensions.keys(),
        key=euclidean_distance,
    )
    return valid_dimensions[closest]


class SensorAndQueue:
    def __init__(self, sensor: Sensor, q: dai.DataOutputQueue):
        self.sensor = sensor
        self.queue = q


class YDNConfigAndQueue:
    def __init__(self, ydn_config: YDNConfig, q: dai.DataOutputQueue):
        self.ydn_config = ydn_config
        self.queue = q


class Worker:
    """
    oak.py <-> worker.py <-> DepthAI SDK <-> DepthAI API (C++ core) <-> actual camera
    """

    logger: Logger

    cfg: OakConfig
    ydn_configs: Mapping[str, YDNConfig]
    user_wants_pc: bool

    pipeline: Optional[dai.Pipeline]
    device: Optional[dai.Device]
    color_sensor_queues: Optional[List[SensorAndQueue]]
    depth_queue: Optional[dai.DataOutputQueue]
    pc_queue: Optional[dai.DataOutputQueue]
    ydn_config_queues: List[YDNConfigAndQueue]

    should_exec: bool
    configured: bool
    running: bool

    depth_stream_name = "depth"
    pc_stream_name = "pc"

    def __init__(
        self,
        oak_config: OakConfig,
        ydn_configs: Mapping[str, YDNConfig],
        user_wants_pc: bool,
    ) -> None:
        self.logger = getLogger(oak_config.name)
        self.logger.warning("Initializing worker.")
        self.cfg = oak_config
        self.ydn_configs = ydn_configs
        self.user_wants_pc = user_wants_pc

        self.device = None
        self.pipeline = None
        self.color_sensor_queues = None
        self.depth_queue = None
        self.pc_queue = None
        self.ydn_config_queues = None

        # Flag for stopping execution and busy loops
        self.should_exec = True
        # Execution status
        self.configured = False
        self.running = False

    def configure(self):
        def configure_color() -> Optional[List[dai.node.ColorCamera]]:
            """
            Creates and configures color cameras or doesn't
            (based on the config).
            """
            cams = []
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

                # Define source and output
                color_cam = self.pipeline.create(dai.node.ColorCamera)
                xout_color = self.pipeline.create(dai.node.XLinkOut)

                xout_color.setStreamName(sensor.get_unique_name())

                # Properties
                color_cam.setResolution(resolution)
                color_cam.setPreviewSize(sensor.width, sensor.height)
                color_cam.setInterleaved(sensor.interleaved)
                color_cam.setBoardSocket(sensor.socket)
                color_cam.setFps(sensor.frame_rate)
                if sensor.color_order == "bgr":
                    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
                else:
                    color_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
                if sensor.manual_focus is not None:
                    color_cam.initialControl.setManualFocus(sensor.manual_focus)

                # Linking
                color_cam.preview.link(xout_color.input)

                cams.append(color_cam)
            return cams

        def configure_stereo() -> Optional[dai.node.StereoDepth]:
            """
            Creates and configures the pipeline for stereo depth— or doesn't
            (based on the config).
            """

            def make_mono_cam(sensor: Sensor) -> dai.node.MonoCamera:
                resolution = get_closest_dai_resolution(
                    sensor.width, sensor.height, DIMENSIONS_TO_MONO_RES
                )
                self.logger.debug(
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
                self.logger.debug("Creating stereo depth component.")
                mono1, mono2 = stereo_pair
                mono_cam_1 = make_mono_cam(mono1)
                mono_cam_2 = make_mono_cam(mono2)

                depth = self.pipeline.create(dai.node.StereoDepth)
                mono_cam_1.out.link(depth.left)
                mono_cam_2.out.link(depth.right)

                # Configuration settings
                depth.setDefaultProfilePreset(
                    dai.node.StereoDepth.PresetMode.HIGH_DENSITY
                )
                depth.setExtendedDisparity(True)
                depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
                depth.initialConfig.setConfidenceThreshold(220)
                depth.initialConfig.setDepthUnit(
                    dai.RawStereoDepthConfig.AlgorithmControl.DepthUnit.MILLIMETER
                )

                xout_depth = self.pipeline.create(dai.node.XLinkOut)
                xout_depth.setStreamName(self.depth_stream_name)
                depth.depth.link(xout_depth.input)

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

        def configure_ydn(color_nodes: List[dai.node.ColorCamera]):
            """
            Creates and configures yolo detection networks or doesn't
            (based on the YDN configs)
            """
            for ydn_config in self.ydn_configs.values():
                nn_out = self.pipeline.create(dai.node.XLinkOut)
                nn_out.setStreamName(f"{ydn_config.service_name}_stream")

                detection_net = self.pipeline.create(dai.node.YoloDetectionNetwork)
                detection_net.setConfidenceThreshold(ydn_config.confidence_threshold)
                detection_net.setNumClasses(len(ydn_config.labels))
                detection_net.setCoordinateSize(ydn_config.coordinate_size)
                detection_net.setAnchors(ydn_config.anchors)
                detection_net.setAnchorMasks(anchorMasks=ydn_config.anchor_masks)
                detection_net.setIouThreshold(ydn_config.iou_threshold)
                detection_net.setNumInferenceThreads(ydn_config.num_threads)
                detection_net.setNumNCEPerInferenceThread(ydn_config.num_nce_per_thread)
                detection_net.setBlobPath(ydn_config.blob_path)
                detection_net.input.setBlocking(False)

                input_node = None
                if ydn_config.input_source == "color":
                    input_node = color_nodes[0]  # primary color cam
                else:  # cam_{n}
                    for color_node in color_nodes:
                        if (
                            color_node.getBoardSocket().name.lower()
                            == ydn_config.input_source
                        ):
                            input_node = color_node
                            break
                if input_node is None:
                    raise ViamError(
                        f'"input_source" could not be matched to a socket or camera output type. Value: {ydn_config.input_source}'
                    )

                input_node.preview.link(detection_net.input)
                detection_net.out.link(nn_out.input)

        self.logger.info(
            f"Num ydn_configs during configuration: {len(self.ydn_configs)}"
        )
        self.pipeline = dai.Pipeline()
        try:
            stage = "color"
            color_nodes = configure_color()

            stage = "stereo"
            depth_node = configure_stereo()

            stage = "point cloud"
            color_node = None
            if color_nodes:
                color_node = color_nodes[0]
            configure_pc(color_node, depth_node)

            stage = "yolo detection network"
            configure_ydn(color_nodes)

            self.logger.info("Successfully configured pipeline.")
        except Exception as e:
            msg = f"Error configuring pipeline at stage '{stage}'. Note the error following this log"
            resolution_err_substr = "bigger than maximum at current sensor resolution"
            calibration_err_substr = "no Camera data available"
            if resolution_err_substr in str(e):
                msg += ". Please adjust 'height_px' and 'width_px' in your config to an accepted resolution."
            elif calibration_err_substr in str(e):
                msg += ". If using a non-integrated model, please check that the camera is calibrated properly."
            self.logger.error(msg)
            raise e

        self.message_synchronizer = MessageSynchronizer()
        self.configured = True

    async def start(self):
        if self.pipeline is None:
            self.logger.error(
                "Cannot start worker: pipeline is None. Did configure() fail?"
            )
            raise ViamError("Cannot start worker: pipeline is None")

        self.device = None
        while not self.device and self.should_exec:
            try:
                device_info_str = self.cfg.device_info
                if device_info_str is None:
                    self.device = dai.Device(pipeline=self.pipeline)
                else:
                    self.device = dai.Device(
                        pipeline=self.pipeline,
                        devInfo=dai.DeviceInfo(mxidOrName=device_info_str),
                    )
                self.device.startPipeline()
                self.logger.info("Successfully initialized device.")
                try:
                    device_info = self.device.getDeviceInfo()
                    self.logger.info(
                        f"Connected device info - Name: {device_info.name}, MxId: {device_info.getMxId()}"
                    )
                except Exception as e:
                    self.logger.error(f"Error getting device info: {e}")

            except Exception as e:
                self.logger.error(f"Error initializing device: {e}")
                await asyncio.sleep(1)

        self.color_sensor_queues: List[SensorAndQueue] = []
        for cs in self.cfg.sensors.color_sensors:
            q = self.device.getOutputQueue(
                cs.get_unique_name(), MAX_COLOR_DEPTH_QUEUE_SIZE, blocking=False
            )
            self.color_sensor_queues.append(SensorAndQueue(cs, q))

        if self.cfg.sensors.stereo_pair:
            self.depth_queue = self.device.getOutputQueue(
                self.depth_stream_name, MAX_COLOR_DEPTH_QUEUE_SIZE, blocking=False
            )
            if self.user_wants_pc:
                self.pc_queue = self.device.getOutputQueue(
                    self.pc_stream_name, MAX_PC_QUEUE_SIZE, blocking=False
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

        self.ydn_config_queues: List[YDNConfigAndQueue] = []
        for ydn_config in self.ydn_configs.values():
            q = self.device.getOutputQueue(
                f"{ydn_config.service_name}_stream", MAX_YDN_QUEUE_SIZE, blocking=False
            )
            self.ydn_config_queues.append(YDNConfigAndQueue(ydn_config, q))

        self.running = True
        self.logger.info("Successfully started camera worker.")

    async def get_synced_color_depth_data(self) -> Tuple[CapturedData, CapturedData]:
        if not self.running:
            raise ViamError(
                "Error getting camera output: Worker is not currently running."
            )
        if len(self.color_sensor_queues) != 1 or not self.depth_queue:
            raise ViamError(
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

            self.logger.debug("Waiting for synced color and depth frames...")
            await asyncio.sleep(0.001)

    async def get_color_output(self, requested_sensor: Sensor):
        if not self.running:
            raise ViamError(
                "Error getting color frame: Worker is not currently running."
            )
        if requested_sensor.sensor_type != "color":
            raise ViamError(
                "Error getting color frame: requested sensor is not a color sensor"
            )

        for sensor_and_queue in self.color_sensor_queues:
            this_sensor = sensor_and_queue.sensor
            if requested_sensor.socket_str != this_sensor.socket_str:
                continue

            q = sensor_and_queue.queue
            msg = None
            while msg is None and self.should_exec:
                try:
                    msg = q.tryGet()
                except Empty:
                    self.logger.debug("Couldn't get color frame: frame queue is empty.")
                await asyncio.sleep(0.01)

            frame = msg.getCvFrame()
            if this_sensor.color_order == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = msg.getTimestamp().total_seconds()
            return CapturedData(frame, timestamp)

        raise ViamError(
            "Error getting color frame: requested sensor not registered in worker"
        )

    async def get_depth_output(self):
        if not self.running:
            raise ViamError(
                "Error getting depth frame: Worker is not currently running."
            )
        if not self.depth_queue:
            raise ViamError(
                "Error getting depth frame: stereo depth frame queue is not configured"
            )

        msg = None
        while msg is None and self.should_exec:
            try:
                msg = self.depth_queue.tryGet()
            except Empty:
                self.logger.debug("Couldn't get depth frame: frame queue is empty.")
            await asyncio.sleep(0.01)

        depth_output = self._process_depth_frame(
            self.cfg.sensors.stereo_pair[0], msg.getCvFrame()
        )
        timestamp = msg.getTimestamp().total_seconds()
        return CapturedData(depth_output, timestamp)

    async def get_pcd(self) -> CapturedData:
        if not self.user_wants_pc:
            raise ViamError(
                "Critical logic error getting PCD: get_pcd() called without toggling user_wants_pc. This is likely a bug."
            )

        attempts = 0
        while (not self.pc_queue or not self.running) and attempts < 10:
            attempts += 1
            self.logger.debug(
                f"Waiting for PCD queue to be initialized. pc_queue: {self.pc_queue}, running: {self.running}, attempts: {attempts}/10"
            )
            await asyncio.sleep(1)
        if not self.pc_queue or not self.running:
            raise ViamError(
                "Error getting PCD: waited, but worker was not ready to get PCD."
            )

        msg: Optional[dai.MessageGroup] = None
        while msg is None and self.should_exec:
            try:
                msg = self.pc_queue.tryGet()
            except Empty:
                self.logger.debug("Couldn't get point cloud: queue is empty.")
            await asyncio.sleep(0.01)

        message_names = msg.getMessageNames()

        if "pcl" not in message_names:
            raise ViamError("Point cloud data not found in message")

        def process_points_and_rgb(
            points: NDArray, color_frame: Optional[NDArray], timestamp: float
        ) -> CapturedData:
            # Apply right-handed conversion if specified in config
            if isinstance(self.cfg, OakDConfig) and self.cfg.right_handed_system:
                points[:, 1] = -points[:, 1]

            if color_frame is not None:
                # Convert BGR to RGB and reshape color data to match points
                rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                colors = (rgb_frame.reshape(-1, 3) / 255.0).astype(np.float64)
                # Convert from 0-1 range to 0-255 range before stacking
                colors_255 = (colors * 255).astype(np.float64)
                points = np.column_stack((points, colors_255))

            return CapturedData(
                self._downsample_pcd(points, MAX_GRPC_MESSAGE_BYTE_COUNT), timestamp
            )

        points = msg["pcl"].getPoints().astype(np.float64)
        timestamp = msg["pcl"].getTimestamp().total_seconds()
        if "rgb" not in message_names:
            color_frame = None
        else:
            color_frame = msg["rgb"].getCvFrame()
        return process_points_and_rgb(points, color_frame, timestamp)

    def get_detections(
        self, service_id: str, service_name: str
    ) -> Optional[dai.ImgDetections]:
        q: Optional[dai.DataOutputQueue] = None
        for obj in self.ydn_config_queues:
            if obj.ydn_config.service_id == service_id:
                q = obj.queue
                break
        if q is None:
            raise ViamError(
                f'Could not find matching YDN config for YDN service id: "{service_id}" and name: "{service_name}"'
            )

        dets = q.tryGet()
        if dets is not None and not isinstance(dets, dai.ImgDetections):
            raise ViamError(f"Incorrect type received for detections: {type(dets)}")
        return dets

    def stop(self) -> None:
        """
        Handles closing resources and exiting logic in worker.
        """
        self.logger.debug("Stopping worker.")
        self.should_exec = False
        self.configured = False
        self.running = False
        if self.device:
            self.device.close()
            self.device = None
        if self.pipeline:
            self.pipeline = None

    def reset(self) -> None:
        self.logger.debug("Resetting worker.")
        self.stop()
        self.should_exec = True

    def _process_depth_frame(self, sensor: Sensor, arr: NDArray) -> NDArray:
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)

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
        if byte_count <= MAX_GRPC_MESSAGE_BYTE_COUNT:
            return arr

        # Get max number of points that can fit within the gRPC message limit using the num bytes per point
        point_size = arr.shape[1] * arr.itemsize
        max_points = MAX_GRPC_MESSAGE_BYTE_COUNT // point_size

        self.logger.warn(
            f"PCD bytes ({byte_count}) > max gRPC bytes count ({MAX_GRPC_MESSAGE_BYTE_COUNT}). Randomly sampling to {max_points} points."
        )

        if arr.ndim == 2:
            indices = np.random.choice(arr.shape[0], max_points, replace=False)
            arr = arr[indices, :]
        else:
            raise ValueError(f"Unexpected point cloud array dimensions: {arr.ndim}")

        return arr


class MessageSynchronizer:
    """
    MessageSynchronizer manages synchronization of frame messages for color and depth data packet queues,
    maintaining an ordered dictionary of messages keyed chronologically by sequence number.
    """

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
            raise ViamError(
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
        while len(self.msgs) > MAX_MSG_SYCHRONIZER_MSGS_SIZE:
            self.msgs.popitem(last=False)  # remove oldest item
