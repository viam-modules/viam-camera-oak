from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional
from typing_extensions import Self

from viam.media.video import CameraMimeType
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
from viam.utils import ValueTypes

from src.config import YDNConfig

LOGGER = getLogger(__name__)


class YoloDetectionNetwork(Vision, Reconfigurable):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "luxonis"), "yolo-detection-network"
    )

    @classmethod
    def new(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        self_obj = cls(config.name)
        self_obj.reconfigure(config, dependencies)
        return self_obj

    # Validates JSON Configuration and returns list of dep names
    @classmethod
    def validate(cls, config: ServiceConfig) -> Sequence[str]:
        return YDNConfig.validate(config.attributes.fields)
    
    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.cfg = YDNConfig(config.attributes.fields)
        self.cfg.initialize_config()
        LOGGER.info(f"Native YDN config: {vars(self.cfg)}")

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
        raise NotImplementedError

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Mapping[str, Any],
        timeout: float,
    ) -> List[Detection]:
        raise NotImplementedError

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Mapping[str, Any],
    ) -> List[Classification]:
        return NotImplementedError

    async def get_classifications_from_camera(self) -> List[Classification]:
        return NotImplementedError

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Mapping[str, Any], timeout: float
    ) -> List[Detection]:
        raise NotImplementedError

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        raise NotImplementedError
    
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
