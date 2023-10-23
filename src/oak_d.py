import logging
from typing import ClassVar, Mapping, Sequence, Any, Dict, Optional, Tuple, Final, List, cast, NamedTuple, Union
from typing_extensions import Self

from PIL import Image

from viam.media.video import NamedImage
from viam.proto.common import ResponseMetadata

from viam.components.camera import DistortionParameters, IntrinsicParameters, RawImage

from viam.module.types import Reconfigurable, Stoppable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from viam.components.camera import Camera
from viam.logging import getLogger

from src.worker import Worker

LOGGER = getLogger(__name__)
DEFAULT_INPUT_WIDTH = 1920
DEFAULT_INPUT_HEIGHT = 1080
DEFAULT_INPUT_FRAMERATE = 30
DEFAULT_IMAGE_MIMETYPE = "image/jpeg"
DEFAULT_DEBUG = False

class OakDModel(Camera, Reconfigurable, Stoppable):
    """
    OakDModel represents a physical OAK-D camera that can capture frames.
    """
    class Properties(NamedTuple):
        intrinsic_parameters: IntrinsicParameters
        """The properties of the camera"""
        distortion_parameters: DistortionParameters
        """The distortion parameters of the camera"""
        supports_pcd: bool = True
        """Whether the camera has a valid implementation of ``get_point_cloud``"""
    

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "camera"), "oak-d")
    worker: Worker
    """``worker`` handles DepthAI integration in a separate thread"""
    camera_properties: Camera.Properties

    # Constructor
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        camera_cls = cls(config.name)
        camera_cls.validate(config)
        camera_cls.reconfigure(config, dependencies)
        return camera_cls

    # Validates JSON Configuration
    @classmethod
    def validate(cls, config: ComponentConfig):
        # here we validate config, the following is just an example and should be updated as needed
        # some_pin = config.attributes.fields["some_pin"].number_value
        # if some_pin == "":
        #     raise Exception("A some_pin must be defined")
        return

    # Handles attribute reconfiguration
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.validate(config)
        try:
            self.worker.stop()
            LOGGER.info("Stopping worker and reconfiguring...")
        except AttributeError:
            self.camera_properties = Camera.Properties(
                supports_pcd=False,
                distortion_parameters=None,
                intrinsic_parameters=None
            )
            LOGGER.info("Starting initial configuration...")

        self.debug = bool(config.attributes.fields["debug"].bool_value) or DEFAULT_DEBUG
        if self.debug:
            LOGGER.setLevel(logging.DEBUG)
            LOGGER.debug(f"Running module in debug mode.")
        self.height_px = int(config.attributes.fields["height_px"].number_value) or DEFAULT_INPUT_HEIGHT
        LOGGER.debug(f'Set height attr to {self.height_px}')
        self.width_px = int(config.attributes.fields["width_px"].number_value) or DEFAULT_INPUT_WIDTH
        LOGGER.debug(f'Set width attr to {self.width_px}')
        self.fps = float(config.attributes.fields["fps"].number_value) or DEFAULT_INPUT_FRAMERATE
        LOGGER.debug(f'Set fps attr to {self.fps}')

        self.worker = Worker(self.height_px, self.width_px, self.fps, self.debug, logger=LOGGER)
        self.worker.start()
        LOGGER.info("Successfully reconfigured!")
        return

    # Implements ``stop`` under the Stoppable protocol to free resources
    def stop(self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs):
        self.worker.stop()

    """ TODO: Implement the methods the Viam RDK defines for the Camera API (rdk:component:camera) """
    async def get_image(
        self, mime_type: str = "", *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Union[Image.Image, RawImage]:
        """Get the next image from the camera as an Image or RawImage.
        Be sure to close the image when finished.

        NOTE: If the mime type is ``image/vnd.viam.dep`` you can use :func:`viam.media.video.RawImage.bytes_to_depth_array`
        to convert the data to a standard representation.

        Args:
            mime_type (str): The desired mime type of the image. This does not guarantee output type

        Returns:
            Image | RawImage: The frame
        """
        LOGGER.debug("Handling get_image request.")
        return self.worker.get_image()

    
    async def get_images(self, *, timeout: Optional[float] = None, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        """Get simultaneous images from different imagers, along with associated metadata.
        This should not be used for getting a time series of images from the same imager.

        Returns:
            Tuple[List[NamedImage], ResponseMetadata]:
                - List[NamedImage]:
                  The list of images returned from the camera system.

                - ResponseMetadata:
                  The metadata associated with this response
        """
        

    
    async def get_point_cloud(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[bytes, str]:
        """
        Get the next point cloud from the camera. This will be
        returned as bytes with a mimetype describing
        the structure of the data. The consumer of this call
        should encode the bytes into the formatted suggested
        by the mimetype.

        To deserialize the returned information into a numpy array, use the Open3D library.
        ::

            import numpy as np
            import open3d as o3d

            data, _ = await camera.get_point_cloud()

            # write the point cloud into a temporary file
            with open("/tmp/pointcloud_data.pcd", "wb") as f:
                f.write(data)
            pcd = o3d.io.read_point_cloud("/tmp/pointcloud_data.pcd")
            points = np.asarray(pcd.points)

        Returns:
            bytes: The pointcloud data.
            str: The mimetype of the pointcloud (e.g. PCD).
        """
        ...

    
    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Properties:
        """
        Get the camera intrinsic parameters and camera distortion parameters

        Returns:
            Properties: The properties of the camera
        """
        return self.camera_properties
