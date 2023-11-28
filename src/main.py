import asyncio

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.oak_d import OakDModel

import viam


Registry.register_resource_creator(
    Camera.SUBTYPE,
    OakDModel.MODEL,
    ResourceCreatorRegistration(OakDModel.new, OakDModel.validate),
)

LOGGER = getLogger(__name__)


async def main():
    """This function creates and starts a new module, after adding all desired resources.
    Resources must be pre-registered. For an example, see the `__init__.py` file.
    """
    module = Module.from_args()
    module.add_model_from_registry(Camera.SUBTYPE, OakDModel.MODEL)
    LOGGER.debug("Starting module in main.py.")
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
