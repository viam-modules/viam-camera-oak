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
        disparity_multiplier = self._add_stereo_depth_node_to(pipeline)

        with dai.Device(pipeline) as device:
            while self.running:
                q_rgb = device.getOutputQueue(RGB_STREAM_NAME)
                rgb_frame_data = q_rgb.tryGet()
                if rgb_frame_data:
                    bgr_frame = rgb_frame_data.getCvFrame()  # OpenCV uses BGR color order
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    self._set_current_image(rgb_frame)

                q_disparity = device.getOutputQueue(DISPARITY_STREAM_NAME)
                disparity_frame = q_disparity.tryGet()
                if disparity_frame:
                    # Apply color map for better visualization
                    frame_disparity = disparity_frame.getCvFrame()
                    # Disparity range is used for normalization
                    frame_disparity = (frame_disparity * disparity_multiplier).astype(np.uint8)
                    frame_disparity = cv2.applyColorMap(frame_disparity, cv2.COLORMAP_JET)
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

    def _add_stereo_depth_node_to(self, pipeline: dai.Pipeline) -> float:
        self.logger.debug("Creating stereo depth node.")
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left = pipeline.create(dai.node.MonoCamera)
        depth = pipeline.create(dai.node.StereoDepth)
        manip = pipeline.create(dai.node.ImageManip)

        disparity_out = pipeline.create(dai.node.XLinkOut)
        disparity_out.setStreamName(DISPARITY_STREAM_NAME)
        manip_out = pipeline.create(dai.node.XLinkOut)
        manip_out.setStreamName(MANIP_STREAM_NAME)
        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName(RIGHT_STREAM_NAME)


        mono_right.setCamera(RIGHT_STREAM_NAME)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setCamera(LEFT_STREAM_NAME)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        depth.setRectifyEdgeFillColor(0)  # Black, to better see the cutout

        # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip.initialConfig.setResize(300, 300)

        mono_right.out.link(xout_right.input)
        mono_right.out.link(depth.right)
        mono_left.out.link(depth.left)
        depth.disparity.link(disparity_out.input)
        depth.rectifiedRight.link(manip.inputImage)
        manip.out.link(manip_out.input)

        return 255 / depth.initialConfig.getMaxDisparity()  # Disparity multiplier is used for normalization

    def _set_current_image(self, arr):
        prev_image, self.current_image = self.current_image, Image.fromarray(arr)
        if prev_image:
            prev_image.close()

    def _set_current_depth_map(self, frame_disparity):
        self.current_depth_map = frame_disparity.tobytes()
        self.logger.debug(f"type of depth map: {type(self.current_depth_map)}")
        self.logger.debug(f"length of depth map: {len(self.current_depth_map)}")
