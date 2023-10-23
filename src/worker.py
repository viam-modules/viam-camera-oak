from logging import Logger
from threading import Thread

import depthai
import cv2
import numpy as np
from PIL import Image

RGB_STREAM_NAME = "rgb"

# TODO: add logging and debug logging throughout
class Worker(Thread):
    height: int
    width: int
    fps: float
    debug: bool
    logger: Logger

    def __init__(self, height: int, width: int, fps: float, debug: bool, logger: Logger) -> None:
        logger.debug("Initializing worker.")

        self.height = height
        self.width = width
        self.fps = fps
        self.debug = debug
        self.logger = logger

        self.running = False
        self.current_image = None
        self.running = True
        super().__init__()
 
    def get_image(self):
        if self.current_image:
            return self.current_image.copy()
        return None

    def run(self) -> None:
        self.logger.info("Initializing worker's image pipeline.")
        pipeline = depthai.Pipeline()

        # TODO: add nodes here and refactor pipeline process
        self._add_camera_rgb_node_to(pipeline)

        with depthai.Device(pipeline) as device:
            while self.running:
                rgb_stream_queue = device.getOutputQueue(RGB_STREAM_NAME)  # Keep inside loop in case of camera reconnects
                rgb_frame_data = rgb_stream_queue.tryGet()
                if not rgb_frame_data:
                    continue  # Reset iteration until data is successfully dequeued
                
                bgr_frame = rgb_frame_data.getCvFrame()  # As of 10/23/2023, OpenCV uses BGR color order
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                prev_image, self.current_image = self.current_image, Image.fromarray(rgb_frame)
                if prev_image:
                    prev_image.close()

        self.logger.info("Worker thread finished.")

    def stop(self) -> None:
        self.running = False
    
    def _add_camera_rgb_node_to(self, pipeline: depthai.Pipeline) -> depthai.DataOutputQueue:
        self.logger.debug("Creating color camera")
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_rgb.setVideoSize(self.width, self.height)
        cam_rgb.setFps(self.fps)
        cam_rgb.setInterleaved(False)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName(RGB_STREAM_NAME)
        cam_rgb.video.link(xout_rgb.input)