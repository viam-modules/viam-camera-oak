"""
This file registers the model with the Python SDK.
"""

from viam.components.camera import Camera
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .oakD import oakD

Registry.register_resource_creator(Camera.SUBTYPE, oakD.MODEL, ResourceCreatorRegistration(oakD.new, oakD.validate))
