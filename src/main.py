import asyncio

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.vision import Vision

from src.components.oak import Oak
from src.services.yolo_detection_network import (
    YoloDetectionNetwork,
)

LOGGER = getLogger("viam-luxonis-module-logger")


def register_oak(module: Module) -> None:
    for model in Oak.ALL_MODELS:
        Registry.register_resource_creator(
            Camera.SUBTYPE,
            model,
            ResourceCreatorRegistration(Oak.new, Oak.validate),
        )
        module.add_model_from_registry(Camera.SUBTYPE, model)


def register_yolo(module: Module) -> None:
    model = YoloDetectionNetwork.MODEL
    Registry.register_resource_creator(
        Vision.SUBTYPE,
        model,
        ResourceCreatorRegistration(
            YoloDetectionNetwork.new, YoloDetectionNetwork.validate
        ),
    )
    module.add_model_from_registry(Vision.SUBTYPE, model)


async def main():
    """This function creates and starts a new module, after adding all desired resources.
    Resources must be pre-registered. For an example, see the `__init__.py` file.
    """
    module = Module.from_args()
    register_oak(module)
    register_yolo(module)

    LOGGER.debug("Starting module in main.py.")
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
