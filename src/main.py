import asyncio

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.module.module import Module
from src.oak_d import OakDModel


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
