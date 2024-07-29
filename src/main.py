import asyncio

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.components.oak import Oak


LOGGER = getLogger("oak-main-logger")


async def main():
    """This function creates and starts a new module, after adding all desired resources.
    Resources must be pre-registered. For an example, see the `__init__.py` file.
    """
    for model in Oak.ALL_MODELS:
        Registry.register_resource_creator(
            Camera.SUBTYPE,
            model,
            ResourceCreatorRegistration(Oak.new, Oak.validate),
        )

    module = Module.from_args()
    for model in Oak.ALL_MODELS:
        module.add_model_from_registry(Camera.SUBTYPE, model)
    LOGGER.debug("Starting module in main.py.")
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
