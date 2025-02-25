import asyncio
from logging import Logger
from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional
from typing_extensions import Self
import uuid

from viam.errors import ViamError
from viam.components.camera import Camera
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision
from viam.media.video import ViamImage
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.logging import getLogger
from viam.services.vision import CaptureAllResult

from src.config import YDNConfig
from src.do_command_helpers import (
    encode_ydn_configure_command,
    encode_ydn_deconfigure_command,
    decode_detections,
    decode_capture_all_result,
    YDN_CAPTURE_ALL,
)


class YoloDetectionNetwork(Vision, Reconfigurable):
    """
    YoloDetectionNetwork is the implementation of the yolo detection network
    service model in this module. It wraps the OAK component and sends do_commands
    to (re)configure the underlying pipeline to support on-VPU YOLO detection inference.
    It serves Vision detection APIs and capture all.
    """

    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "luxonis"), "yolo-detection-network"
    )
    cam_name: str
    cam: Camera
    ydn_service_name: str
    ydn_service_id: uuid.UUID
    pipeline_configured: bool
    should_exec: bool
    logger: Logger

    @classmethod
    def new(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        self = cls(config.name)
        self.logger = getLogger(config.name)
        self.ydn_service_id = uuid.uuid4()
        self.pipeline_configured = False
        self.should_exec = True
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate(cls, config: ServiceConfig) -> Sequence[str]:
        logger = getLogger("viam-ydn-validation")
        return YDNConfig.validate(config.attributes.fields, logger)

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """
        This reconfigure method reinitializes the config and updates the YDNConfig
        on the component pipeline side using a do_command.

        Args:
            config (ServiceConfig): Viam service config
            dependencies (Mapping[ResourceName, ResourceBase]): list of deps

        Raises:
            e: KeyError if camera dep is not found in dependencies
        """
        self.cfg = YDNConfig(config.attributes.fields, config.name)
        self.cfg.initialize_config()

        self.cam_name = config.attributes.fields["cam_name"].string_value
        try:
            self.cam = dependencies[Camera.get_resource_name(self.cam_name)]
        except KeyError as e:
            self.logger.error(
                f"Could not find camera dep '{self.cam_name}' in deps list."
            )
            raise e
        self.ydn_service_name = config.name

        async def async_do_command():
            attempts = 0
            max_attempts = 5
            while (
                not self.pipeline_configured
                and self.should_exec
                and attempts < max_attempts
            ):
                try:
                    configure_cmd = encode_ydn_configure_command(
                        self.cfg, self.ydn_service_name, self.ydn_service_id
                    )
                    await self.cam.do_command(command=configure_cmd)
                    self.pipeline_configured = True
                except Exception as e:
                    self.logger.warn(
                        f"Error in do_command: {e}. Trying to configure via do_command again..."
                    )
                    attempts += 1
                    if attempts >= max_attempts:
                        self.logger.error(
                            "Max attempts reached. Failed to configure pipeline in OAK component for YOLO Detection Network."
                        )
                await asyncio.sleep(1)

        asyncio.create_task(async_do_command())

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        if camera_name == "" or camera_name is None:
            self.logger.warn(
                f"camera_name arg was unspecified. Defaulting to '{self.cam_name}'"
            )
        elif camera_name != self.cam_name:
            raise ViamError(
                f'Requested camera name "{camera_name}" does not match configured OAK camera "{self.cam_name}".'
            )

        if return_classifications:
            raise ViamError(
                "Classifications are not supported in the YOLO detection network service."
            )
        if return_object_point_clouds:
            raise ViamError(
                "Object point clouds are not supported in the YOLO detection network service."
            )
        if return_image == False and return_detections == False:
            raise ViamError("Please request either return_image or return_detections.")

        d = await self.cam.do_command(
            {
                "cmd": YDN_CAPTURE_ALL,
                "return_detections": return_detections,
                "return_image": return_image,
                "sender_id": self.ydn_service_id.hex,
                "sender_name": self.ydn_service_name,
            }
        )
        return decode_capture_all_result(d)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Mapping[str, Any],
        timeout: float,
    ) -> List[Detection]:
        self.logger.warn(
            "WARNING! get_detections calls get_detections_from_camera under the hood. Use only for debugging purposes. Your image will be ignored."
        )
        return await self.get_detections_from_camera("", extra=None, timeout=None)

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Mapping[str, Any], timeout: float
    ) -> List[Detection]:
        if camera_name == "" or camera_name is None:
            self.logger.warn(
                f"camera_name arg was unspecified. Defaulting to '{self.cam_name}'"
            )
        elif camera_name != self.cam_name:
            raise ViamError(
                f'Requested camera name "{camera_name}" does not match configured OAK camera "{self.cam_name}".'
            )

        mapping = await self.cam.do_command(
            {
                "cmd": YDN_CAPTURE_ALL,
                "return_detections": True,
                "return_image": False,
                "sender_id": self.ydn_service_id.hex,
                "sender_name": self.ydn_service_name,
            }
        )
        self.logger.info(f"Received dets: {mapping}")
        if "detections" not in mapping:
            raise ViamError(
                "Critical logic error: 'detections' attr not found in response. This is likely a bug."
            )
        return decode_detections(mapping["detections"])

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def close(self) -> None:
        """
        Implements `close` to free resources on shutdown.
        """
        self.should_exec = False
        if self.pipeline_configured:
            deconfigure_cmd = encode_ydn_deconfigure_command(self.ydn_service_id)
            await self.cam.do_command(command=deconfigure_cmd)

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Mapping[str, Any],
    ) -> List[Classification]:
        raise NotImplementedError()

    async def get_classifications_from_camera(self) -> List[Classification]:
        raise NotImplementedError()

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError()
