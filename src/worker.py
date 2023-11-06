import asyncio
from logging import Logger
from threading import Thread
import time
from typing import Callable, Union

import cv2
import depthai as dai
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import PointcloudPacket, DisparityDepthPacket
from depthai_sdk.components.camera_component import CameraComponent
from depthai_sdk.components.pointcloud_component import PointcloudComponent
from depthai_sdk.components.stereo_component import StereoComponent
import numpy as np
from numpy.typing import NDArray

PREVIEW_STREAM_NAME = 'PREVIEW'
MAX_PIPELINE_FAILURES = 3
MAX_GRPC_MESSAGE_BYTE_COUNT = 4194304  # Update this if the gRPC config ever changes
CAM_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P

class WorkerManager(Thread):
    def __init__(self,
                logger: Logger,
                reconfigure: Callable[[None], None],
                ) -> None:
        self.logger = logger
        self.reconfigure = reconfigure

        self.needs_reconfigure = False
        self.running = True
        super().__init__()
    
    def run(self) -> None:
        self.logger.debug('Starting worker manager.')
        while self.running:
            self.logger.debug('Checking if worker must be reconfigured.')
            if self.needs_reconfigure:
                self.logger.debug('Worker needs reconfiguring; reconfiguring worker.')
                self.reconfigure()
            time.sleep(5)

    def stop(self) -> None:
        self.logger.debug('Stopping worker manager.')
        self.running = False

class CapturedData:
    def __init__(self, np_array: NDArray, captured_at: float) -> None:
        self.np_array = np_array
        self.captured_at = captured_at

class Worker(Thread):
    color_image: Union[CapturedData, None]
    depth_map: Union[CapturedData, None]
    pcd: Union[CapturedData, None]
    user_called_pcd: bool

    manager: WorkerManager

    def __init__(self,
                height: int,
                width: int,
                frame_rate: float,
                user_wants_color: bool,
                user_wants_depth: bool,
                reconfigure: Callable[[None], None],
                logger: Logger,
                ) -> None:
        logger.info('Initializing camera pipeline worker.')

        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.user_wants_color = user_wants_color
        self.user_wants_depth = user_wants_depth
        self.logger = logger

        self.color_image = None
        self.depth_map = None
        self.pcd = None
        self.user_called_pcd = False

        self.manager = WorkerManager(logger, reconfigure)
        self.manager.start()
        super().__init__()
 
    async def get_color_image(self) -> CapturedData:
        while not self.color_image and self.manager.running:
            self.logger.debug('Waiting for color image...')
            await asyncio.sleep(1)
        return self.color_image
    
    async def get_depth_map(self) -> CapturedData:
        while not self.depth_map and self.manager.running:
            self.logger.debug('Waiting for depth map...')
            await asyncio.sleep(1)
        return self.depth_map
    
    async def get_pcd(self) -> CapturedData:
        if not self.user_called_pcd:
            self.user_called_pcd = True
        while not self.pcd and self.manager.running:
            self.logger.debug('Waiting for pcd...')
            await asyncio.sleep(1)
        return self.pcd

    def _pipeline_loop(self) -> None:
        failures = 0
        try:
            self.logger.debug('Initializing worker image pipeline.')
            with OakCamera() as oak:
                color = self._configure_color(oak)
                stereo = self._configure_stereo(oak, color)
                self._configure_pc(oak, stereo, color)
                prev_user_called_pcd = self.user_called_pcd
                oak.start()
                while self.manager.running:
                    if prev_user_called_pcd != self.user_called_pcd:
                        break  # to restart pipeline with PCD support
                    self._handle_color_output(oak)
                    self._handle_depth_and_pcd_output(oak)
        except Exception as e:
            failures += 1
            if failures > MAX_PIPELINE_FAILURES:
                self.manager.needs_reconfigure = True
                self.logger.error(f"Reached {MAX_PIPELINE_FAILURES} max failures in pipeline loop. Error: {e}")
            else:
                self.logger.debug(f"Pipeline loop failure count: {failures}. Error: {e}. ")
        finally:
            self.logger.debug('Exiting worker camera loop.')

    def run(self) -> None:
        try:
            while self.manager.running:
                self._pipeline_loop()
        finally:
            self.logger.info('Stopped and exited worker thread.')

    def stop(self) -> None:
        self.logger.info('Stopping worker.')
        self.manager.stop()
    
    def _configure_color(self, oak: OakCamera) -> Union[CameraComponent, None] :
        if self.user_wants_color:
            self.logger.debug('Creating pipeline node: color camera.')
            xout_color = oak.pipeline.create(dai.node.XLinkOut)
            xout_color.setStreamName(PREVIEW_STREAM_NAME)
            color = oak.camera('color', fps=self.frame_rate, resolution=CAM_RESOLUTION)
            # setPreviewSize sets the closest supported resolution; inputted height width may not be respected
            color.node.setPreviewSize(self.width, self.height)
            color.node.preview.link(xout_color.input)
            return color

    def _configure_stereo(self, oak: OakCamera, color: CameraComponent) -> Union[StereoComponent, None]:
        if self.user_wants_depth:
            self.logger.debug('Creating pipeline node: stereo depth.')
            # TODO: Find some way for depthai to adjust the output size of depth maps
            # Right now it's being handled as a cv2.resize in _set_depth_map
            # The below commented code should fix, but config_camera for mono camera resizing is not implemented yet
            # mono_left = oak.camera(dai.CameraBoardSocket.LEFT, fps=self.frame_rate)
            # mono_left.config_camera((self.width, self.height))
            # mono_right = oak.camera(dai.CameraBoardSocket.RIGHT, fps=self.frame_rate)
            # mono_right.config_camera((self.width, self.height))
            # stereo = oak.stereo(fps=self.frame_rate, left=mono_left, right=mono_right)
            stereo = oak.stereo(fps=self.frame_rate, resolution='max')
            if self.user_wants_color:
                stereo.config_stereo(align=color)  # ensures alignment and output resolution are same
                stereo.node.setOutputSize(*color.node.getPreviewSize())
            oak.callback(stereo, callback=self._set_depth_map)
            return stereo

    def _configure_pc(self, oak: OakCamera, stereo: StereoComponent, color: CameraComponent) -> Union[PointcloudComponent, None]:
        if self.user_wants_depth and self.user_called_pcd:
            self.logger.debug('Creating pipeline node: point cloud.')
            pcc = oak.create_pointcloud(stereo, color)
            oak.callback(pcc, callback=self._set_pcd)
            return pcc
    
    def _handle_color_output(self, oak: OakCamera) -> None:
        if self.user_wants_color:
            # Get frames from preview stream, not direct out, for correct height and width
            rgb_queue = oak.device.getOutputQueue(PREVIEW_STREAM_NAME)
            rgb_frame_data = rgb_queue.tryGet()
            if rgb_frame_data:
                bgr_frame = rgb_frame_data.getCvFrame()  # OpenCV uses reversed (BGR) color order
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                self._set_color_image(rgb_frame)
    
    def _handle_depth_and_pcd_output(self, oak: OakCamera) -> None:
        if self.user_wants_depth:
            oak.poll()

    def _set_color_image(self, arr: NDArray) -> None:
        self.logger.debug(f'Setting current_image. Array shape: {arr.shape}. Dtype: {arr.dtype}')
        self.color_image = CapturedData(arr, time.time())

    def _set_depth_map(self, packet: DisparityDepthPacket) -> None:
        arr = packet.frame
        if arr.shape != (self.height, self.width):
            self.logger.debug(f'Pipeline output shape mismatch: {arr.shape}; Manually resizing to {(self.height, self.width)}.')
            arr = cv2.resize(arr, (self.width, self.height))
        self.logger.debug(f'Setting current depth map. Array shape: {arr.shape}. Dtype: {arr.dtype}')
        self.depth_map = CapturedData(arr, time.time())

    def _set_pcd(self, packet: PointcloudPacket) -> None:
        arr, num_bytes = packet.points, packet.points.nbytes
        self.logger.debug(f'Setting current pcd. num_bytes: {num_bytes}')
        if num_bytes > MAX_GRPC_MESSAGE_BYTE_COUNT:
            self.logger.warn(f'PCD bytes ({num_bytes}) > max gRPC bytes count ({MAX_GRPC_MESSAGE_BYTE_COUNT}). Subsampling data 0.5x.')
            arr = arr[::2, ::2, :]
        self.pcd = CapturedData(arr, time.time())
