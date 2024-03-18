"""
This file registers the model with the Python SDK.
"""

from viam.components.camera import Camera
from viam.resource.registry import Registry, ResourceCreatorRegistration

from src.oak import Oak


for model in Oak.MODELS:
    Registry.register_resource_creator(
        Camera.SUBTYPE,
        model,
        ResourceCreatorRegistration(Oak.new, Oak.validate),
    )
