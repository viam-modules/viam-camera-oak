from logging import Logger
from threading import Thread
import time
from typing import Callable

import cv2
import depthai as dai
from depthai_sdk import OakCamera
from numpy.typing import NDArray

RGB_STREAM_NAME = "rgb"
DEPTH_STREAM_NAME = "depth"
RIGHT_STREAM_NAME = "right"
LEFT_STREAM_NAME = "left"

class WorkerManager(Thread):
    def __init__(self,
                debug: bool,
                logger: Logger,
                reconfigure: Callable[[None], None],
                ) -> None:
        self.debug = debug
        self.logger = logger
        self.needs_reconfigure = False
        self.running = True
        self.reconfigure = reconfigure
        super().__init__()
    
    def run(self):
        self.logger.info("Starting worker status manager.")
        while self.running:
            self.logger.debug("Checking if worker must be reconfigured.")
            if self.needs_reconfigure:
                self.logger.debug("Worker needs reconfiguring; reconfiguring worker.")
                self.reconfigure()
            time.sleep(1)

    def stop(self):
        self.logger.info("Stopping worker status manager.")
        self.running = False


class Worker(Thread):
    current_image: NDArray
    current_depth_map: NDArray
    manager: WorkerManager

    def __init__(self,
                height: int,
                width: int,
                frame_rate: float,
                debug: bool,
                logger: Logger,
                reconfigure: Callable[[None], None],
                ) -> None:
        logger.debug("Initializing worker.")

        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.debug = debug
        self.logger = logger

        self.current_image = None
        self.current_depth_map = None

        self.manager = WorkerManager(debug, logger, reconfigure)
        self.manager.start()
        super().__init__()
 
    def get_current_image(self):
        return self.current_image
    
    def get_current_depth_map(self):
        return self.current_depth_map

    def _pipeline_loop(self):
        try:
            self.logger.debug("Initializing worker image pipeline.")
            with OakCamera() as oak:
                self._add_camera_rgb_node(oak)
                self._add_depth_node(oak)
                oak.start()
                while self.manager.running:
                    rgb_queue = oak.device.getOutputQueue(RGB_STREAM_NAME)
                    rgb_frame_data = rgb_queue.tryGet()
                    if rgb_frame_data:
                        bgr_frame = rgb_frame_data.getCvFrame()  # OpenCV uses reversed (BGR) color order
                        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        self._set_current_image(rgb_frame)

                    q_depth = oak.device.getOutputQueue(DEPTH_STREAM_NAME, maxSize=4, blocking=False)
                    depth_frame = q_depth.tryGet()
                    if depth_frame:
                        np_depth_arr = depth_frame.getCvFrame()
                        self._set_current_depth_map(np_depth_arr)
        except Exception as e:
            self.manager.needs_reconfigure = True
            self.logger.error(e)
        finally:
            self.logger.debug("Exiting worker camera loop.")

    def run(self) -> None:
        try:
            while self.manager.running:
                self._pipeline_loop()
        finally:
            self.logger.info("Exiting worker thread.")
        

    def stop(self) -> None:
        self.logger.info("Stopping worker.")
        self.manager.stop()
    
    def _add_camera_rgb_node(self, oak: OakCamera):
        self.logger.debug("Creating pipeline node: color camera.")
        xout_color = oak.pipeline.create(dai.node.XLinkOut)
        xout_color.setStreamName(RGB_STREAM_NAME)
        color = oak.camera("color", resolution=None, fps=self.frame_rate)
        color.node.video.link(xout_color.input)

    def _add_depth_node(self, oak: OakCamera):
        self.logger.debug("Creating pipeline node: depth.")
        mono_right = oak.pipeline.create(dai.node.MonoCamera)
        mono_left = oak.pipeline.create(dai.node.MonoCamera)

        depth_out = oak.pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName(DEPTH_STREAM_NAME)
        xout_right = oak.pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName(RIGHT_STREAM_NAME)
        xout_left = oak.pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName(LEFT_STREAM_NAME)

        stereo = oak.stereo(resolution=None, fps=self.frame_rate, left=mono_left, right=mono_right)
        stereo.node.depth.link(depth_out.input)

    def _set_current_image(self, np_arr):
        self.logger.debug("Setting current_image.")
        self.current_image = np_arr

    def _set_current_depth_map(self, np_arr):
        self.logger.debug(f"Setting current depth map.")
        self.current_depth_map = np_arr
