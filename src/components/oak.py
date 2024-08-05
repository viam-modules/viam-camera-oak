import asyncio
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)
from typing_extensions import Self

# Viam module
from viam.errors import NotSupportedError, ViamError
from viam.logging import getLogger
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

# Viam camera
from viam.components.camera import (
    Camera,
    DistortionParameters,
    IntrinsicParameters,
)
from viam.media.video import CameraMimeType, NamedImage, ViamImage

# OAK module
from src.components.worker.worker import Worker
from src.components.helpers.shared import CapturedData
from src.components.helpers.encoders import (
    encode_jpeg_bytes,
    encode_depth_raw,
    encode_pcd,
    handle_synced_color_and_depth,
    make_metadata_from_seconds_float,
)
from src.components.helpers.config import (
    Validator,
    OakConfig,
    OakDConfig,
    OakFfc3PConfig,
)
from src.components.worker.worker_manager import WorkerManager


LOGGER = getLogger("viam-oak-module-logger")

# Be sure to update README.md if default attributes are changed
DEFAULT_IMAGE_MIMETYPE = CameraMimeType.JPEG


class Oak(Camera, Reconfigurable):
    """
    This class implements all available methods for the camera class: get_image,
    get_images, get_point_cloud, and get_properties.

    It inherits from the built-in resource subtype Base and conforms to the
    ``Reconfigurable`` protocol, which signifies that this component can be
    reconfigured.

    The constructor conforms to the ``resource.types.ResourceCreator``
    type required for all models.
    """

    class Properties(NamedTuple):
        intrinsic_parameters: IntrinsicParameters
        """The properties of the camera"""
        distortion_parameters: DistortionParameters
        """The distortion parameters of the camera"""
        supports_pcd: bool = True
        """Whether the camera has a valid implementation of ``get_point_cloud``"""

    _deprecated_family = ModelFamily("viam", "camera")
    _depr_oak_agnostic_model = Model(_deprecated_family, "oak")
    _depr_oak_d_model = Model(_deprecated_family, "oak-d")
    DEPRECATED_MODELS: ClassVar[Tuple[Model]] = (
        _depr_oak_agnostic_model,
        _depr_oak_d_model,
    )
    _family = ModelFamily("viam", "luxonis")
    _oak_ffc_3p_model = Model(_family, "oak-ffc-3p")
    _oak_d_model = Model(_family, "oak-d")
    SUPPORTED_MODELS: ClassVar[Tuple[Model]] = (
        _oak_ffc_3p_model,
        _oak_d_model,
    )
    ALL_MODELS: ClassVar[Tuple[Model]] = DEPRECATED_MODELS + SUPPORTED_MODELS

    model: Model
    """Viam model of component"""
    oak_cfg: OakConfig
    """Native config"""
    worker: Optional[Worker] = None
    """`Worker` handles camera logic in a separate thread"""
    worker_manager: Optional[WorkerManager] = None
    """`WorkerManager` managing the lifecycle of `worker`"""
    get_point_cloud_was_invoked: bool = False
    camera_properties: Camera.Properties

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """
        Used to register the module model class as a resource.

        Args:
            config (ComponentConfig)
            dependencies (Mapping[ResourceName, ResourceBase])

        Returns:
            Self: the OAK model class
        """
        camera_cls = cls(config.name)
        camera_cls.validate(config)
        camera_cls.reconfigure(config, dependencies)
        return camera_cls

    @classmethod
    def validate(cls, config: ComponentConfig) -> None:
        """
        A procedure called to validate the camera config.

        Args:
            config (ComponentConfig)

        Raises:
            ValidationError: with a description of what is wrong about the config

        Returns:
            None
        """
        validator = Validator(config)

        if config.model == str(cls._depr_oak_agnostic_model):
            LOGGER.warn(
                f"The '{cls._depr_oak_agnostic_model}' is deprecated. Please switch to '{cls._oak_d_model}' or '{cls._oak_ffc_3p_model}'"
            )
            cls.model = cls._oak_d_model
        elif config.model == str(cls._depr_oak_d_model):
            LOGGER.warn(
                f"The '{cls._depr_oak_d_model}' is deprecated. Please switch to '{cls._oak_d_model}'"
            )
            cls.model = cls._oak_d_model
        elif config.model == str(cls._oak_d_model):
            cls.model = cls._oak_d_model
        elif config.model == str(cls._oak_ffc_3p_model):
            cls.model = cls._oak_ffc_3p_model
        else:
            raise ViamError(f"Cannot validate unrecognized model: {cls.model}")

        validator.validate_shared_attrs()
        if cls.model == cls._oak_d_model:
            validator.validate_oak_d()
        else:
            validator.validate_oak_ffc_3p()

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        """
        A procedure both the RDK and module invokes to (re)configure and (re)boot the module—
        serving as an initializer and restart method.

        Args:
            config (ComponentConfig)
            dependencies (Mapping[ResourceName, ResourceBase])
        """
        self._close()

        if self.model == self._oak_d_model:
            self.oak_cfg = OakDConfig(config)
        elif self.model == self._oak_ffc_3p_model:
            self.oak_cfg = OakFfc3PConfig(config)
        else:
            raise ViamError(
                f"Critical error due to spec change of validation failure: unrecognized model {self.model}"
            )

        supports_pcd = bool(self.oak_cfg.sensors.stereo_pair)
        self.camera_properties = Camera.Properties(
            supports_pcd=supports_pcd,
            distortion_parameters=None,
            intrinsic_parameters=None,
        )

        self.worker = Worker(
            oak_config=self.oak_cfg,
            user_wants_pc=self.get_point_cloud_was_invoked,
        )

        self.worker_manager = WorkerManager(self.worker)
        self.worker_manager.start()

    async def close(self) -> None:
        """
        Implements `close` to free resources on shutdown.
        """
        LOGGER.info("Closing OAK component.")
        self._close()
        LOGGER.debug("Closed OAK component.")

    def _close(self) -> None:
        if self.worker_manager:
            self.worker_manager.stop()
            self.worker_manager.join()
        if self.worker:
            self.worker.stop()

    async def get_image(
        self,
        mime_type: str = "",
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ViamImage:
        """
        Gets the next image from the camera as an Image or RawImage.
        Be sure to close the image when finished.

        NOTE: If the mime type is ``image/vnd.viam.dep`` you can use :func:`viam.media.video.RawImage.bytes_to_depth_array`
        to convert the data to a standard representation.

        Args:
            mime_type (str): The desired mime type of the image. This does not guarantee output type.

        Raises:
            NotSupportedError: if mime_type is not supported for the method
            ViamError: if validation somehow failed and no sensors are configured

        Returns:
            PIL.Image.Image | RawImage: The frame
        """
        mime_type = self._validate_get_image_mime_type(mime_type)

        await self._wait_until_worker_running()

        main_sensor_type = self.oak_cfg.sensors.primary_sensor.sensor_type
        if main_sensor_type == "color":
            if mime_type == CameraMimeType.JPEG:
                arr = self.worker.get_color_output(
                    self.oak_cfg.sensors.primary_sensor
                ).np_array
                jpeg_encoded_bytes = encode_jpeg_bytes(arr)
                return ViamImage(jpeg_encoded_bytes, CameraMimeType.JPEG)

            raise NotSupportedError(
                f'mime_type "{mime_type}" is not supported for color. Please use {CameraMimeType.JPEG}'
            )

        if main_sensor_type == "depth":
            captured_data = self.worker.get_depth_output()
            arr = captured_data.np_array
            if mime_type == CameraMimeType.JPEG:
                jpeg_encoded_bytes = encode_jpeg_bytes(arr, is_depth=True)
                return ViamImage(jpeg_encoded_bytes, CameraMimeType.JPEG)
            if mime_type == CameraMimeType.VIAM_RAW_DEPTH:
                depth_encoded_bytes = encode_depth_raw(arr.tobytes(), arr.shape)
                return ViamImage(depth_encoded_bytes, CameraMimeType.VIAM_RAW_DEPTH)
            raise NotSupportedError(
                f"mime_type {mime_type} is not supported for depth. Please use {CameraMimeType.JPEG} or {CameraMimeType.VIAM_RAW_DEPTH}."
            )

        raise ViamError(
            'get_image failed due to misconfigured "sensors" attribute, but should have been validated in `validate`...'
        )

    async def get_images(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[List[NamedImage], ResponseMetadata]:
        """
        Gets images from every sensor on your OAK device.

        Returns:
            Tuple[List[NamedImage], ResponseMetadata]:
                - List[NamedImage]:
                  The list of images returned from the camera system.

                - ResponseMetadata:
                  The metadata associated with this response
        """
        LOGGER.debug("get_images called")

        await self._wait_until_worker_running()

        # Split logic into helpers for OAK-D and OAK-D like cameras with FFC-like cameras
        # Use MessageSynchronizer only for OAK-D-like
        if self.oak_cfg.sensors.color_sensors and self.oak_cfg.sensors.stereo_pair:
            color_data, depth_data = await self.worker.get_synced_color_depth_data()
            return handle_synced_color_and_depth(color_data, depth_data)

        images: List[NamedImage] = []
        # For timestamp calculation later
        seconds_float: float = None

        if self.oak_cfg.sensors.color_sensors:
            for cs in self.oak_cfg.sensors.color_sensors:
                color_data: CapturedData = self.worker.get_color_output(cs)
                arr, captured_at = color_data.np_array, color_data.captured_at
                jpeg_encoded_bytes = encode_jpeg_bytes(arr)
                img = NamedImage("color", jpeg_encoded_bytes, CameraMimeType.JPEG)
                seconds_float = captured_at
                images.append(img)

        if self.oak_cfg.sensors.stereo_pair:
            depth_data: CapturedData = self.worker.get_depth_output()
            arr, captured_at = depth_data.np_array, depth_data.captured_at
            depth_encoded_bytes = encode_depth_raw(arr.tobytes(), arr.shape)
            img = NamedImage(
                "depth", depth_encoded_bytes, CameraMimeType.VIAM_RAW_DEPTH
            )
            seconds_float = captured_at
            images.append(img)

        metadata = make_metadata_from_seconds_float(seconds_float)
        return images, metadata

    async def get_point_cloud(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Tuple[bytes, str]:
        """
        Gets the next point cloud from the camera. This will be
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
            with open('/tmp/pointcloud_data.pcd', 'wb') as f:
                f.write(data)
            pcd = o3d.io.read_point_cloud('/tmp/pointcloud_data.pcd')
            points = np.asarray(pcd.points)

        Raises:
            MethodNotAllowed: when config doesn't supply "depth" as a sensor

        Returns:
            bytes: The serialized point cloud data.
            str: The mimetype of the point cloud (e.g. PCD).
        """
        # Validation
        if not self.oak_cfg.sensors.stereo_pair:
            details = "Cannot process PCD. OAK camera not configured for stereo depth outputs. See README for details"
            raise MethodNotAllowed(method_name="get_point_cloud", details=details)

        await self._wait_until_worker_running()

        # By default, we do not get point clouds even when color and depth are both requested
        # We have to reinitialize the worker/pipeline+device to start making point clouds
        if not self.worker.user_wants_pc:
            self.worker.user_wants_pc = True
            self.worker.reset()
            self.worker.configure()
            await self.worker.start()

        while not self.worker.running:
            LOGGER.info("Waiting for worker to restart with pcd configured...")
            await asyncio.sleep(0.5)

        # Get actual PCD data from camera worker
        pcd_obj = self.worker.get_pcd()
        arr = pcd_obj.np_array

        return encode_pcd(arr)

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Properties:
        """
        Gets the camera intrinsic parameters and camera distortion parameters

        Returns:
            Properties: The properties of the camera
        """
        return self.camera_properties

    async def _wait_until_worker_running(self, max_attempts=5, timeout_seconds=1):
        """
        Blocks on camera data methods that require the worker to be running.
        Unblocks once worker is running or max number of attempts to pass is reached.

        Args:
            max_attempts (int, optional): Defaults to 5.
            timeout_seconds (int, optional): Defaults to 1.

        Raises: ViamError
        """
        attempts = 0
        while attempts < max_attempts:
            if self.worker.running:
                return
            attempts += 1
            await asyncio.sleep(timeout_seconds)
        raise ViamError(
            "Camera data requested before camera worker was ready. Please ensure the camera is properly "
            "connected and configured, especially for non-integrated models such as the OAK-FFC."
        )

    def _validate_get_image_mime_type(
        self, mime_type: CameraMimeType
    ) -> CameraMimeType:
        """
        Validates inputted mime type for get_image calls.

        Args:
            mime_type (CameraMimeType): user inputted mime_type

        Raises:
            err: NotSupportedError

        Returns:
            CameraMimeType: validated/converted mime type
        """
        # Guard for empty str (no inputted mime_type)
        if mime_type == "":
            return CameraMimeType.JPEG  # default is JPEG

        # Get valid types based on main sensor
        main_sensor_type = self.oak_cfg.sensors.primary_sensor.sensor_type
        if main_sensor_type == "color":
            valid_mime_types = [CameraMimeType.JPEG]
        else:  # depth
            valid_mime_types = [CameraMimeType.JPEG, CameraMimeType.VIAM_RAW_DEPTH]

        # Check validity
        if mime_type not in valid_mime_types:
            err = NotSupportedError(
                f'mime_type "{mime_type}" is not supported for get_image.'
                f"Valid mime type(s): {valid_mime_types}."
            )
            LOGGER.error(err)
            raise err

        return mime_type


class MethodNotAllowed(ViamError):
    """
    Exception raised when attempting to call a method
    with a configuration that does not support said method.
    """

    def __init__(self, method_name: str, details: str) -> None:
        self.name = method_name
        self.message = (
            f'Cannot invoke method "{method_name}" with current config. {details}'
        )
        super().__init__(self.message)
