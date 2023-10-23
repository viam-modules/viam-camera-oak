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
        self.logger.info("Initializing camera pipeline")
        pipeline = depthai.Pipeline()

        # add nodes here
        self._add_camera_rgb_node(pipeline)

        frame = None
        with depthai.Device(pipeline) as device:
            while self.running:
                q_rgb = device.getOutputQueue(RGB_STREAM_NAME)
                in_rgb = q_rgb.tryGet()
                if in_rgb is None:
                    continue
                
                frame = in_rgb.getCvFrame()
                if frame is None:
                    self.logger.warn("Could not convert message to CvFrame")
                    continue

                self.lock.acquire()
                try:
                    f_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    prev_image, self.current_image = self.current_image, Image.fromarray(f_tmp)
                    if prev_image is not None:
                        try:
                            prev_image.close()
                        except:
                            pass
                    # TODO: except with possible errors (idk what they are rn)
                finally:
                    self.lock.release()

        self.logger.info("Stopped worker thread.")

    def stop(self) -> None:
        self.running = False

    def _debug_log(self, msg):
        if self.debug:
            self.logger.debug(msg)
    
    def _add_camera_rgb_node(self, pipeline: depthai.Pipeline) -> depthai.DataOutputQueue:
        self._debug_log("Creating color camera")
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_rgb.setVideoSize(self.width, self.height)
        cam_rgb.setFps(self.fps)
        cam_rgb.setInterleaved(False)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName(RGB_STREAM_NAME)
        cam_rgb.video.link(xout_rgb.input)