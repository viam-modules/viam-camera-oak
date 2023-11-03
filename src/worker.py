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
    color_image: CapturedData
    depth_map: CapturedData
    pcd: CapturedData
    manager: WorkerManager

    def __init__(self,
                height: int,
                width: int,
                frame_rate: float,
                should_get_color: bool,
                should_get_depth: bool,
                reconfigure: Callable[[None], None],
                logger: Logger,
                ) -> None:
        logger.info('Initializing camera pipeline worker.')

        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.should_get_color = should_get_color
        self.should_get_depth = should_get_depth
        self.logger = logger

        self.color_image = CapturedData(None, None)
        self.depth_map = CapturedData(None, None)
        self.pcd = CapturedData(None, None)

        self.manager = WorkerManager(logger, reconfigure)
        self.manager.start()
        super().__init__()
 
    def get_color_image(self) -> CapturedData:
        return self.color_image
    
    def get_depth_map(self) -> CapturedData:
        return self.depth_map
    
    def get_pcd(self) -> CapturedData:
        return self.pcd

    def _pipeline_loop(self) -> None:
        failures = 0
        try:
            self.logger.debug('Initializing worker image pipeline.')
            with OakCamera() as oak:
                color = self._add_camera_rgb_node(oak)
                stereo = self._add_depth_node(oak, color)
                self._add_pc_node(oak, color, stereo)
                oak.start()
                while self.manager.running:
                    self._handle_color_output(oak)
                    self._handle_depth_and_pcd_output(oak)
        except Exception as e:
            failures += 1
            if failures > MAX_PIPELINE_FAILURES:
                self.manager.needs_reconfigure = True
                self.logger.error(f"Exceeded {MAX_PIPELINE_FAILURES} max failures on pipeline loop. Error: {e}")
            else:
                self.logger.debug(f"Pipeline failure count: {failures}: Error: {e}")
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
    
    def _add_camera_rgb_node(self, oak: OakCamera) -> Union[CameraComponent, None] :
        if self.should_get_color:
            self.logger.debug('Creating pipeline node: color camera.')
            xout_color = oak.pipeline.create(dai.node.XLinkOut)
            xout_color.setStreamName(PREVIEW_STREAM_NAME)
            color = oak.camera('color', fps=self.frame_rate, resolution=CAM_RESOLUTION)
            # setPreviewSize sets the closest supported resolution; inputted height width may not be respected
            color.node.setPreviewSize(self.width, self.height)
            color.node.preview.link(xout_color.input)
            return color

    def _add_depth_node(self, oak: OakCamera, color: CameraComponent) -> Union[StereoComponent, None]:
        if self.should_get_depth:
            self.logger.debug('Creating pipeline node: stereo depth.')
            stereo = oak.stereo(fps=self.frame_rate, resolution='max')
            if self.should_get_color:
                stereo.config_stereo(align=color)  # ensures alignment and output resolution are same
                stereo.node.setOutputSize(*color.node.getPreviewSize())
            else:
                # TODO: find some way for depthai to adjust the output size of depth maps
                # right now it's being handled as a cv2.resize in _set_depth_map
                pass
            oak.callback(stereo, callback=self._set_depth_map)
            return stereo

    def _add_pc_node(self, oak: OakCamera, color: CameraComponent, stereo: StereoComponent) -> Union[PointcloudComponent, None]:
        if self.should_get_depth:
            self.logger.debug('Creating pipeline node: point cloud.')
            pcc = oak.create_pointcloud(stereo=stereo, colorize=color)
            oak.callback(pcc, callback=self._set_pcd)
            return pcc
    
    def _handle_color_output(self, oak: OakCamera) -> None:
        if self.should_get_color:
            # Get frames from preview stream, not direct out, for correct height and width
            rgb_queue = oak.device.getOutputQueue(PREVIEW_STREAM_NAME)
            rgb_frame_data = rgb_queue.tryGet()
            if rgb_frame_data:
                bgr_frame = rgb_frame_data.getCvFrame()  # OpenCV uses reversed (BGR) color order
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                self._set_color_image(rgb_frame)
    
    def _handle_depth_and_pcd_output(self, oak: OakCamera) -> None:
        if self.should_get_depth:
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
        self.logger.debug('Setting current pcd.')
        subsampled_points = packet.points[::2, ::2, :]
        self.pcd = CapturedData(subsampled_points, time.time())
