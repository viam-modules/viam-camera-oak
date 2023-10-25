from logging import Logger
from threading import Thread

import depthai as dai
import cv2
import numpy as np
from PIL import Image

RGB_STREAM_NAME = "rgb"
DISPARITY_STREAM_NAME = "disparity"
RIGHT_STREAM_NAME = "right"
LEFT_STREAM_NAME = "left"
MANIP_STREAM_NAME = "manip"

# TODO: add logging and debug logging throughout
class Worker(Thread):
    height: int
    width: int
    frame_rate: float
    debug: bool
    logger: Logger

    def __init__(self, height: int, width: int, frame_rate: float, debug: bool, logger: Logger) -> None:
        logger.debug("Initializing worker.")

        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        self.debug = debug
        self.logger = logger

        self.running = False
        self.current_image = None
        self.current_depth_map = None
        self.running = True
        super().__init__()
 
    def get_current_image(self):
        if self.current_image:
            return self.current_image.copy()
        return None
    
    def get_current_depth_map(self):
        return self.current_depth_map

    def run(self) -> None:
        self.logger.info("Initializing worker's image pipeline.")
        pipeline = dai.Pipeline()

        self._add_camera_rgb_node_to(pipeline)
        disparity_multiplier = self._add_depth_node_to(pipeline)
        with dai.Device(pipeline) as device:
            while self.running:
                q_rgb = device.getOutputQueue(RGB_STREAM_NAME)
                rgb_frame_data = q_rgb.tryGet()
                if rgb_frame_data:
                    bgr_frame = rgb_frame_data.getCvFrame()  # OpenCV uses BGR color order
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    self._set_current_image(rgb_frame)

                qDepth = device.getOutputQueue(name=DISPARITY_STREAM_NAME, maxSize=4, blocking=False)
                depth_frame = qDepth.get()
                if depth_frame:
                    frame_disparity = depth_frame.getCvFrame()
                    frame_disparity = (frame_disparity * disparity_multiplier).astype(np.uint8)
                    self._set_current_depth_map(frame_disparity)
        self.logger.info("Worker thread finished.")

    def stop(self) -> None:
        self.running = False
    
    def _add_camera_rgb_node_to(self, pipeline: dai.Pipeline):
        self.logger.debug("Creating color camera node.")
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setVideoSize(self.width, self.height)
        cam_rgb.setFps(self.frame_rate)
        cam_rgb.setInterleaved(False)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName(RGB_STREAM_NAME)
        cam_rgb.video.link(xout_rgb.input)
    
    def _add_depth_node_to(self, pipeline: dai.Pipeline):
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)

        disparityOut = pipeline.create(dai.node.XLinkOut)
        disparityOut.setStreamName(DISPARITY_STREAM_NAME)
        xoutRight = pipeline.create(dai.node.XLinkOut)
        xoutRight.setStreamName('right')

        monoRight.setCamera("right")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.setRectifyEdgeFillColor(0) # Black, to better see the cutout

        monoRight.out.link(depth.right)
        monoLeft.out.link(depth.left)
        depth.disparity.link(disparityOut.input)

        # Disparity range is used for normalization
        return 255 / depth.initialConfig.getMaxDisparity()

    def _set_current_image(self, arr):
        prev_image, self.current_image = self.current_image, Image.fromarray(arr)
        if prev_image:
            prev_image.close()
        self.logger.debug("Setting current_image.")

    def _set_current_depth_map(self, np_array):
        self.current_depth_map = self.current_depth_map, np_array
        self.logger.debug(f"Setting current depth map.")
