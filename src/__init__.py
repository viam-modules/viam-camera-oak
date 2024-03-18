"""
This file registers the model with the Python SDK.
"""

from viam.components.camera import Camera
from viam.resource.registry import Registry, ResourceCreatorRegistration

from src.oak import Oak

Registry.register_resource_creator(
    Camera.SUBTYPE,
    Oak.MODEL,
    ResourceCreatorRegistration(Oak.new, Oak.validate),
)
