import asyncio
import math
import time
from typing import Any, Callable, Mapping, Tuple, Union

from logging import Logger
from threading import Thread

import cv2
import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import (
    DisparityDepthPacket,
    FramePacket,
    PointcloudPacket,
)
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.pointcloud_component import PointcloudComponent
from depthai_sdk.components.stereo_component import StereoComponent
from numpy.typing import NDArray


DIMENSIONS_TO_MONO_RES = {
    (1280, 800): dai.MonoCameraProperties.SensorResolution.THE_800_P,
    (1280, 720): dai.MonoCameraProperties.SensorResolution.THE_720_P,
    (640, 480): dai.MonoCameraProperties.SensorResolution.THE_480_P,
    (640, 400): dai.MonoCameraProperties.SensorResolution.THE_400_P,
}  # stereo camera component only accepts this subset of depthai_sdk.components.camera_helper.monoResolutions

DIMENSIONS_TO_COLOR_RES = {
    (4056, 3040): dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    (3840, 2160): dai.ColorCameraProperties.SensorResolution.THE_4_K,
    (1920, 1080): dai.ColorCameraProperties.SensorResolution.THE_1080_P,
}  # color camera component only accepts this subset of depthai_sdk.components.camera_helper.colorResolutions


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


class CapturedData:
    """
    CapturedData is image data with the data as an np array,
    plus the time.time() it was captured at.
    """

    def __init__(self, np_array: NDArray, captured_at: float) -> None:
        self.np_array = np_array
        self.captured_at = captured_at


class Worker:
    """
    OakDModel <-> Worker <-> DepthAI SDK <-> DepthAI API (C++ core) <-> actual OAK-D
    """

    color_image: Union[CapturedData, None]
    depth_map: Union[CapturedData, None]
    pcd: Union[CapturedData, None]

    def __init__(
        self,
        height: int,
        width: int,
        frame_rate: float,
        user_wants_color: bool,
        user_wants_depth: bool,
        reconfigure: Callable[[None], None],
        logger: Logger,
    ) -> None:
        logger.info("Initializing worker.")

        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.user_wants_color = user_wants_color
        self.user_wants_depth = user_wants_depth
        self.reconfigure = reconfigure
        self.logger = logger

        self.color_image = None
        self.depth_map = None
        self.pcd = None
        self.running = True

        self._init_oak_camera()
        self._config_oak_camera()

        self.manager = WorkerManager(self.oak, logger, reconfigure)
        self.manager.start()

    async def get_color_image(self) -> CapturedData:
        while not self.color_image and self.running:
            self.logger.debug("Waiting for color image...")
            await asyncio.sleep(0.01)
        return self.color_image

    async def get_depth_map(self) -> CapturedData:
        while not self.depth_map and self.running:
            self.logger.debug("Waiting for depth map...")
            await asyncio.sleep(0.01)
        return self.depth_map

    async def get_pcd(self) -> CapturedData:
        while not self.pcd and self.running:
            self.logger.debug("Waiting for pcd...")
            await asyncio.sleep(0.01)
        return self.pcd

    def stop(self) -> None:
        """
        Handles stopping and closing of the camera worker.
        """
        self.logger.info("Stopping worker.")
        self.manager.stop()
        self.oak.close()

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
            stage = "stereo"
            stereo = self._configure_stereo(color)
            stage = "point cloud"
            self._configure_pc(stereo, color)
            stage = "start"
            self.oak.start()
        except Exception as e:
            err_str = f"Error configuring OakCamera at stage '{stage}': {e}"
            resolution_err_substr = "bigger than maximum at current sensor resolution"
            if type(e) == RuntimeError and resolution_err_substr in str(e):
                err_str += ". Please adjust 'height_px' and 'width_px' in your config to an accepted resolution."
            self.logger.error(err_str)

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

    def _configure_color(self) -> Union[CameraComponent, None]:
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
            self.oak.callback(color, callback=self._set_color_image)
            return color
        return None

    def _configure_stereo(
        self, color: Union[CameraComponent, None]
    ) -> Union[StereoComponent, None]:
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
            cam_left = CameraComponent(
                self.oak.device,
                self.oak.pipeline,
                dai.CameraBoardSocket.CAM_B,  # Same as CameraBoardSocket.LEFT
                dai.MonoCameraProperties.SensorResolution.THE_400_P,
                self.frame_rate,
            )
            cam_right = CameraComponent(
                self.oak.device,
                self.oak.pipeline,
                dai.CameraBoardSocket.CAM_C,  # Same as CameraBoardSocket.RIGHT
                dai.MonoCameraProperties.SensorResolution.THE_400_P,
                self.frame_rate,
            )
            stereo = self.oak.stereo(resolution, self.frame_rate, cam_left, cam_right)
            if color:
                # Ensures camera alignment and output resolution are same for color and depth
                stereo.config_stereo(align=color)
                stereo.node.setOutputSize(*color.node.getPreviewSize())
            self.oak.callback(stereo, callback=self._set_depth_map)
            return stereo
        return None

    def _configure_pc(
        self, stereo: Union[StereoComponent, None], color: Union[CameraComponent, None]
    ) -> Union[PointcloudComponent, None]:
        """
        Creates and configures point cloud component— or doesn't
        (based on the config)

        Returns:
            Union[PointcloudComponent, None]
        """
        if self.user_wants_color and self.user_wants_depth:
            self.logger.debug("Creating point cloud component.")
            pcc = self.oak.create_pointcloud(stereo, color)
            self.oak.callback(pcc, callback=self._set_pcd)
            return pcc
        return None

    def _set_color_image(self, packet: FramePacket) -> None:
        """
        Passed as a callback func to DepthAI to use when new color data is outputted.
        Callback logic chronologically syncs all the outputted data.

        Args:
            packet (FramePacket): outputted color data inputted by caller
        """
        # DepthAI outputs BGR; convert to RGB
        arr: NDArray = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        self.logger.debug(
            f"Setting color image. Array shape: {arr.shape}. Dtype: {arr.dtype}"
        )
        self.color_image = CapturedData(arr, time.time())

    def _set_depth_map(self, packet: DisparityDepthPacket) -> None:
        """
        Passed as a callback func to DepthAI to use when new depth data is outputted.
        Callback logic chronologically syncs all the outputted data.

        Args:
            packet (DisparityDepthPacket): outputted depth data inputted by caller
        """
        arr = packet.frame
        if arr.shape[0] > self.height and arr.shape[1] > self.width:
            self.logger.debug(
                f"Outputted depth map's shape is greater than specified in config: {arr.shape}; Manually resizing to {(self.height, self.width)}."
            )
            top_left_x = (arr.shape[1] - self.width) // 2
            top_left_y = (arr.shape[0] - self.height) // 2
            arr = arr[
                top_left_y : top_left_y + self.height,
                top_left_x : top_left_x + self.width,
            ]

        self.logger.debug(
            f"Setting depth map. Array shape: {arr.shape}. Dtype: {arr.dtype}"
        )
        self.depth_map = CapturedData(arr, time.time())

    def _set_pcd(self, packet: PointcloudPacket) -> None:
        """
        Passed as a callback func to DepthAI to use when new PCD data is outputted.
        Callback logic chronologically syncs all the outputted data.

        Args:
            packet (PointcloudPacket): outputted PCD data inputted by caller
        """
        arr, byte_count = packet.points, packet.points.nbytes
        self.logger.debug(f"Setting pcd. Byte count: {byte_count}")
        self.pcd = CapturedData(arr, time.time())
