"""
This file registers the model with the Python SDK.
"""

from viam.components.camera import Camera
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily

class Fake:
    def new():
        pass
    def validate():
        pass

Registry.register_resource_creator(
    Camera.SUBTYPE,
    Model(ModelFamily("viam", "camera"), "oak-d"),
    ResourceCreatorRegistration(Fake.new, Fake.validate),
)
