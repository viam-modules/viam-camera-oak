from logging import Logger
from threading import Thread

import depthai as dai
import cv2
import numpy as np
from PIL import Image

RGB_STREAM_NAME = "rgb"
DISPARITY_STREAM_NAME = "disparity"
DEPTH_STREAM_NAME = "depth"
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
        return self.current_image
    
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

                q_depth = device.getOutputQueue(DEPTH_STREAM_NAME, maxSize=4, blocking=False)
                depth_frame = q_depth.tryGet()
                if depth_frame:
                    np_depth_arr = depth_frame.getCvFrame()
                    np_depth_arr = np_depth_arr.astype(np.uint16)
                    self._set_current_depth_map(np_depth_arr)
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
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        # disparity_out = pipeline.create(dai.node.XLinkOut)
        # disparity_out.setStreamName(DISPARITY_STREAM_NAME)
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName(DEPTH_STREAM_NAME)
        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName(RIGHT_STREAM_NAME)
        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName(LEFT_STREAM_NAME)

        mono_right.setCamera("right")
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setCamera("left")
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout

        mono_right.out.link(stereo.right)
        mono_left.out.link(stereo.left)
        # stereo.disparity.link(disparity_out.input)
        stereo.depth.link(depth_out.input)

        # Disparity range is used for normalization
        return 255 / stereo.initialConfig.getMaxDisparity()

    def _set_current_image(self, np_arr):
        self.current_image = np_arr
        self.logger.debug("Setting current_image.")

    def _set_current_depth_map(self, np_arr):
        self.current_depth_map = np_arr
        self.logger.debug(f"Setting current depth map.")
