"""
This file registers the model with the Python SDK.
"""

from viam.components.camera import Camera
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .oak_d import OakDModel

Registry.register_resource_creator(
    Camera.SUBTYPE,
    OakDModel.MODEL,
    ResourceCreatorRegistration(OakDModel.new, OakDModel.validate),
)
