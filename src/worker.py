from logging import Logger
from threading import Thread, Lock

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
        self.lock = Lock()
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
                self.lock.acquire()
                try:
                    # Initialize queue and dequeue data packet
                    rgb_queue = device.getOutputQueue(RGB_STREAM_NAME)
                    rgb_frame_data = rgb_queue.tryGet()
                    if rgb_frame_data is None:
                        self.logger.warn("tryGet image from rgb img queue failed.")
                        continue
                    
                    # Convert frame data into OpenCV compatible frame (BGR space)
                    bgr_frame = rgb_frame_data.getCvFrame()
                    if bgr_frame is None:
                        self.logger.warn("Could not convert message to CvFrame")
                        continue
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

                    # Set current image to PIL.Image of the RGB frame and close the previous image
                    prev_image, self.current_image = self.current_image, Image.fromarray(rgb_frame)
                    if prev_image:
                        prev_image.close()
                finally:
                    self.lock.release()

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