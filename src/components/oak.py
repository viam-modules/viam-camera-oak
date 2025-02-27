# Standard library
import asyncio
from logging import Logger
from threading import Lock, Thread
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)
from typing_extensions import Self

# Viam SDK
from viam.errors import NotSupportedError, ViamError
from viam.logging import getLogger
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes
from viam.components.camera import (
    Camera,
    DistortionParameters,
    IntrinsicParameters,
)
from viam.media.video import CameraMimeType, NamedImage, ViamImage

# Local
from src.components.worker.worker import Worker
from src.components.helpers.shared import CapturedData
from src.components.helpers.encoders import (
    encode_jpeg_bytes,
    encode_depth_raw,
    encode_pcd,
    handle_synced_color_and_depth,
    convert_seconds_float_to_metadata,
)
from src.config import OakConfig, OakDConfig, OakFfc3PConfig, YDNConfig
from src.components.worker.worker_manager import WorkerManager
from src.do_command_helpers import (
    decode_ydn_configure_command,
    encode_detections,
    encode_image_data,
    # YDN = yolo detection network
    YDN_CONFIGURE,
    YDN_DECONFIGURE,
    YDN_CAPTURE_ALL,
)

DEFAULT_IMAGE_MIMETYPE = CameraMimeType.JPEG
RECONFIGURE_TIMEOUT_SECONDS = 0.1


class Oak(Camera, Reconfigurable):
    """
    This class implements all available methods for the camera class: get_image,
    get_images, get_point_cloud, and get_properties. The underlying hardware
    is an OAK family camera supported in the model list.

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
    logger: ClassVar[Logger]
    """Class scoped logger"""
    init_reconfig_manager_lock: ClassVar[Lock] = Lock()
    """Lock for thread-safe access to initializing reconfig_manager"""
    # The 'OakInstanceManager' is put in quotes because it is a forward reference.
    # This is necessary bc OakInstanceManager is not yet defined at the point of its usage.
    # Using quotes allows Python to understand that this is a type hint for a class
    # that will be defined later in the code.
    reconfig_manager: ClassVar[Optional["OakInstanceManager"]] = None  # type: ignore
    """Handler for reconfiguring the camera"""

    logger: Logger
    """Instance scoped logger"""
    model: Model
    """Viam model of component"""
    oak_cfg: OakConfig
    """Native config"""
    ydn_configs: Mapping[str, YDNConfig]
    """Configs populated by ydn service"""
    ydn_configs_lock: Lock
    """Lock for thread-safe access to ydn_configs"""
    worker: Optional[Worker] = None
    """`Worker` handles camera logic in a separate thread"""
    worker_manager: Optional[WorkerManager] = None
    """`WorkerManager` managing the lifecycle of `worker`"""
    camera_properties: Camera.Properties
    """Camera properties as per Viam SDK"""
    is_closed: bool = False
    """Flag to indicate if the camera has been closed by Close()"""

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """
        SDK calls this method to create a new camera instance.
        """
        self = cls(config.name)
        cls.validate(config)
        self.ydn_configs = dict()  # YDN configs have to persist across reconfigures
        self.ydn_configs_lock = Lock()
        cls.logger = getLogger("viam-oak")
        self.logger = getLogger(config.name)

        with cls.init_reconfig_manager_lock:
            if cls.reconfig_manager is None or cls.reconfig_manager.stop_event.is_set():
                cls.reconfig_manager = OakInstanceManager(
                    [(self, config, dependencies)]
                )
                cls.reconfig_manager.start()
            else:
                cls.reconfig_manager.add_reconfigure_request(self, config, dependencies)
        return self

    @classmethod
    def validate(cls, config: ComponentConfig) -> List[str]:
        """
        A procedure called to validate the camera config.

        Args:
            config (ComponentConfig)

        Raises:
            ValidationError: with a description of what is wrong about the config

        Returns:
            List[str]: of dep names
        """
        if config.model == str(cls._depr_oak_agnostic_model):
            cls.logger.warning(
                f"The '{cls._depr_oak_agnostic_model}' is deprecated. Please switch to '{cls._oak_d_model}' or '{cls._oak_ffc_3p_model}'"
            )
            cls.model = cls._oak_d_model
        elif config.model == str(cls._depr_oak_d_model):
            cls.logger.warning(
                f"The '{cls._depr_oak_d_model}' is deprecated. Please switch to '{cls._oak_d_model}'"
            )
            cls.model = cls._oak_d_model
        elif config.model == str(cls._oak_d_model):
            cls.model = cls._oak_d_model
        elif config.model == str(cls._oak_ffc_3p_model):
            cls.model = cls._oak_ffc_3p_model
        else:
            raise ViamError(f"Cannot validate unrecognized model: {cls.model}")

        logger = getLogger("viam-oak-validation")
        if cls.model == cls._oak_d_model:
            return OakDConfig.validate(config.attributes.fields, logger)
        else:
            return OakFfc3PConfig.validate(config.attributes.fields, logger)

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        """
        SDK calls this method to reconfigure the camera without losing connection to the driver.
        This method is used to add the request to the OakInstanceManager's queue.

        Args:
            config (ComponentConfig)
            dependencies (Mapping[ResourceName, ResourceBase])
        """
        cls = type(self)
        cls.reconfig_manager.add_reconfigure_request(self, config, dependencies)

    def do_reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        """
        This method is called by the OakInstanceManager to actually reconfigure the camera.
        """
        self._close()

        if self.model == self._oak_d_model:
            self.oak_cfg = OakDConfig(config.attributes.fields, config.name)
        elif self.model == self._oak_ffc_3p_model:
            self.oak_cfg = OakFfc3PConfig(config.attributes.fields, config.name)
        else:
            raise ViamError(
                f"Critical logic error due to spec change or validation failure: unrecognized model {self.model}. This is likely a bug."
            )
        self.oak_cfg.initialize_config()

        supports_pcd = bool(self.oak_cfg.sensors.stereo_pair)
        self.camera_properties = Camera.Properties(
            supports_pcd=supports_pcd,
            distortion_parameters=None,
            intrinsic_parameters=None,
        )

        with self.ydn_configs_lock:
            self.worker = Worker(
                oak_config=self.oak_cfg,
                ydn_configs=self.ydn_configs,
            )

        self.worker_manager = WorkerManager(self.worker)
        self.worker_manager.start()

    async def close(self) -> None:
        """
        Implements `close` camera method to free resources on shutdown.
        """
        self.logger.info("Closing OAK component.")
        self._close()
        self.is_closed = True
        self.logger.info("Closed OAK component.")

    def _close(self) -> None:
        if self.worker_manager and not self.worker_manager.stop_event.is_set():
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

        await self._wait_for_worker()

        main_sensor_type = self.oak_cfg.sensors.primary_sensor.sensor_type
        if main_sensor_type == "color":
            if mime_type == CameraMimeType.JPEG:
                captured_data = await self.worker.get_color_output(
                    self.oak_cfg.sensors.primary_sensor
                )
                arr = captured_data.np_array
                jpeg_encoded_bytes = encode_jpeg_bytes(arr)
                return ViamImage(jpeg_encoded_bytes, CameraMimeType.JPEG)

            raise NotSupportedError(
                f'mime_type "{mime_type}" is not supported for color. Please use {CameraMimeType.JPEG}'
            )

        if main_sensor_type == "depth":
            captured_data = await self.worker.get_depth_output()
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
        self.logger.debug("get_images called")

        await self._wait_for_worker()

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
                color_data: CapturedData = await self.worker.get_color_output(cs)
                arr, captured_at = color_data.np_array, color_data.captured_at
                jpeg_encoded_bytes = encode_jpeg_bytes(arr)
                img = NamedImage("color", jpeg_encoded_bytes, CameraMimeType.JPEG)
                seconds_float = captured_at
                images.append(img)

        if self.oak_cfg.sensors.stereo_pair:
            depth_data: CapturedData = await self.worker.get_depth_output()
            arr, captured_at = depth_data.np_array, depth_data.captured_at
            depth_encoded_bytes = encode_depth_raw(arr.tobytes(), arr.shape)
            img = NamedImage(
                "depth", depth_encoded_bytes, CameraMimeType.VIAM_RAW_DEPTH
            )
            seconds_float = captured_at
            images.append(img)

        metadata = convert_seconds_float_to_metadata(seconds_float)
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
            details = "Cannot process PCD. OAK camera not configured for stereo depth outputs. Please see module docs in app configuration card."
            raise MethodNotAllowed(method_name="get_point_cloud", details=details)
        if type(self.oak_cfg) == OakDConfig and not self.oak_cfg.point_cloud_enabled:
            raise MethodNotAllowed(
                method_name="get_point_cloud",
                details="attribute 'point_cloud_enabled' is unspecified or set to false. Please set to true in app configuration card.",
            )

        await self._wait_for_worker()

        try:
            pcd_obj = await self.worker.get_pcd()
            arr = pcd_obj.np_array
            return encode_pcd(arr)
        except Exception as e:
            raise ViamError(f"Error getting point cloud data: {e}")

    async def get_properties(
        self, *, timeout: Optional[float] = None, **kwargs
    ) -> Properties:
        """
        Gets the camera intrinsic parameters and camera distortion parameters

        Returns:
            Properties: The properties of the camera
        """
        return self.camera_properties

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        # Get cmd type from command
        if "cmd" not in command:
            raise ViamError(
                "Critical logic error: 'cmd' field not present in OAK's do_command mapping arg. This is likely a bug."
            )
        cmd = command["cmd"]

        # Handle different commands conditionally
        if cmd == YDN_CONFIGURE:
            self.logger.debug(f"Received YDN_CONFIGURE with mapping: {command}")
            await self._wait_for_worker()
            ydn_config = decode_ydn_configure_command(command)
            with self.ydn_configs_lock:
                self.ydn_configs[ydn_config.service_id] = ydn_config
            self.logger.info(
                "Closing camera to reconfigure pipeline with yolo detection network."
            )
            self.worker_manager.restart_atomic_bool.set(True)
            return {}
        elif cmd == YDN_DECONFIGURE:
            self.logger.debug(f"Received YDN_DECONFIGURE with mapping: {command}")
            id = command["sender_id"]
            with self.ydn_configs_lock:
                if id in self.ydn_configs:
                    del self.ydn_configs[id]
            self.worker.reset()
            return {}
        elif cmd == YDN_CAPTURE_ALL:
            await self._wait_for_worker()
            resp = dict()
            service_id = command["sender_id"]
            service_name = command["sender_name"]
            with self.ydn_configs_lock:
                try:
                    ydn_config = self.ydn_configs[service_id]
                except KeyError:
                    raise ViamError(
                        f'Could not find matching YDN config for YDN service id: "{service_id}" and name: "{service_name}"'
                    )

            # Find respective Sensor to the YDN config
            sensor = None
            if ydn_config.input_source == "color":
                try:
                    sensor = self.oak_cfg.sensors.color_sensors[
                        0
                    ]  # primary color sensor
                except (AttributeError, IndexError) as e:
                    self.logger.error(
                        f'"color" input source was requested by service "{ydn_config.service_name}", but no color camera exists in OAK config.'
                    )
                    raise e
            else:  # input_source is like f"cam_{x}"
                for cs in self.oak_cfg.sensors.color_sensors:
                    if ydn_config.input_source == cs.socket_str:
                        sensor = cs
                        break
            if sensor is None:
                self.logger.error(
                    f'"{ydn_config.input_source}" was requested by service "{ydn_config.service_name}", but was not found in the OAK config.'
                )

            if command["return_detections"]:
                detections = self.worker.get_detections(service_id, service_name)
                if detections is None:
                    resp["detections"] = []
                else:
                    resp["detections"] = encode_detections(
                        detections, ydn_config.labels, sensor
                    )
            if command["return_image"]:
                captured_data = await self.worker.get_color_output(sensor)
                arr = captured_data.np_array
                jpeg_encoded_bytes = encode_jpeg_bytes(arr)
                resp["image_data"] = encode_image_data(jpeg_encoded_bytes)
            return resp
        else:
            raise ViamError(f'"cmd": "{cmd}" is not a valid command.')

    async def _wait_for_worker(
        self,
        max_attempts=5,
        timeout_seconds=1,
        desired_status: Literal["running", "configured"] = "running",
    ):
        """
        Blocks on camera data methods that require the worker to be the desired status.
        Unblocks once worker is in the desired status or max number of attempts to pass is reached.

        Args:
            max_attempts (int, optional): Defaults to 5.
            timeout_seconds (int, optional): Defaults to 1.
            desired_status (Literal["running", "configured"], optional): Defaults to "running",

        Raises: ViamError
        """
        if desired_status not in ["running", "configured"]:
            raise ViamError(
                f"Critical logic error: _wait_for_worker was called with unrecognized desired status: {desired_status}. This is likely a bug."
            )

        attempts = 0
        while attempts < max_attempts:
            if self.worker:
                if self.worker.running and desired_status == "running":
                    return
                if self.worker.configured and desired_status == "configured":
                    return
            attempts += 1
            await asyncio.sleep(timeout_seconds)
        raise ViamError(
            f"Camera data requested before camera worker was {desired_status}. Please ensure the camera is properly "
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
            self.logger.error(err)
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


class OakInstanceManager(Thread):
    """
    OakInstanceManager is a thread that manages every Oak instance.
    It is responsible for orchestrating each Oak instance when one or more reconfigure requests are made.
    """

    def __init__(
        self,
        reqs: List[Tuple[Oak, ComponentConfig, Mapping[ResourceName, ResourceBase]]],
    ) -> None:
        super().__init__()
        self.daemon = True  # mark as daemon thread so it doesn't prevent program exit
        self.reconfig_reqs: List[
            Tuple[Oak, ComponentConfig, Mapping[ResourceName, ResourceBase]]
        ] = reqs
        self.lock: Lock = Lock()
        self.requests_waited: bool = False
        self.oak_instances: List[Oak] = []
        self.stop_event: asyncio.Event = asyncio.Event()
        self.logger: Logger = getLogger("oak_manager")

        self.logger.info("OakInstanceManager initialized with requests: %s", reqs)

        self.loop = asyncio.new_event_loop()
        self.loop.create_task(self.handle_reconfigures())

    async def handle_reconfigures(self) -> None:
        self.logger.info("Starting reconfiguration handler loop.")
        while not self.stop_event.is_set():
            await asyncio.sleep(RECONFIGURE_TIMEOUT_SECONDS)

            # Check for reconfiguration requests
            reqs_to_process = None
            with self.lock:
                if not self.reconfig_reqs:
                    continue

                if not self.requests_waited:
                    self.requests_waited = True
                    self.logger.debug(
                        "Waiting one time for more reconfiguration requests."
                    )
                    continue

                # Process requests after waiting one iteration
                reqs_to_process = self.reconfig_reqs
                self.reconfig_reqs = []
                self.requests_waited = False
                self.logger.debug(
                    "Processing reconfiguration requests: %s", reqs_to_process
                )

            # Process reconfiguration requests
            if reqs_to_process:
                has_device_info = False
                # Check if device_info is specified in any of the reconfiguration requests
                for _, config, _ in reqs_to_process:
                    if config.attributes.fields["device_info"].string_value is not None:
                        has_device_info = True
                        break
                if has_device_info:
                    # If device_info is specified, close all existing Oak instances before reconfiguring
                    # This is to ensure that when swapping device_infos, the old device is closed before
                    # the new device is opened. If we don't do this, we may encounter a race condition where
                    # the new device is opened before the old device is closed, causing the new device to fail to open.
                    [oak._close() for oak, _, _ in reqs_to_process]

                for oak, config, dependencies in reqs_to_process:
                    self.logger.debug(
                        "Reconfiguring Oak instance: %s with config: %s", oak, config
                    )
                    oak.do_reconfigure(config, dependencies)
                    if oak not in self.oak_instances:
                        self.oak_instances.append(oak)
                        self.logger.debug("Added Oak instance to active list: %s", oak)

            self.oak_instances = [
                oak for oak in self.oak_instances if not oak.is_closed
            ]
            if len(self.oak_instances) == 0:
                self.logger.info(
                    "No active Oak instances. Stopping reconfiguration loop."
                )
                self.stop_event.set()

    def add_reconfigure_request(
        self,
        oak: Oak,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> None:
        with self.lock:
            self.reconfig_reqs.append((oak, config, dependencies))
            self.logger.debug("Added reconfigure request for Oak instance: %s", oak)
            self.logger.debug("New reconfigure requests list: %s", self.reconfig_reqs)

    def run(self):
        self.logger.debug("Starting OakInstanceManager event loop.")
        try:
            self.loop.run_forever()
        except Exception as e:
            self.logger.error(f"Error in OakInstanceManager loop: {e}")
        finally:
            self.logger.info("Cleaning up OakInstanceManager resources")
            pending_tasks = asyncio.all_tasks(self.loop)
            for task in pending_tasks:
                task.cancel()

            # Close all oak instances
            for oak in self.oak_instances:
                if not oak.is_closed:
                    try:
                        oak._close()
                    except Exception as e:
                        self.logger.error(f"Error closing Oak instance: {e}")

            try:
                self.loop.close()
            except Exception as e:
                self.logger.error(f"Error closing event loop: {e}")
