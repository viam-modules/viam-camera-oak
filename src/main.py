import asyncio
import sys

from viam.module.module import Module
from viam.components.camera import Camera
from .oak_d import OakDModel

async def main():
    '''This function creates and starts a new module, after adding all desired resources.
    Resources must be pre-registered. For an example, see the `__init__.py` file.
    '''
    module = Module.from_args()
    module.add_model_from_registry(Camera.SUBTYPE, OakDModel.MODEL)
    await module.start()

if __name__ == '__main__':
    asyncio.run(main())
