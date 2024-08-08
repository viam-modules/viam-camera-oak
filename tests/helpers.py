from typing import Any, Mapping
from google.protobuf.struct_pb2 import Struct

from viam.proto.app.robot import ComponentConfig, ServiceConfig

def make_component_config(dictionary: Mapping[str, Any], model: str) -> ComponentConfig:
    struct = Struct()
    struct.update(dictionary=dictionary)
    return ComponentConfig(attributes=struct, model=model)

def make_service_config(dictionary: Mapping[str, Any], model: str) -> ServiceConfig:
    struct = Struct()
    struct.update(dictionary=dictionary)
    return ServiceConfig(attributes=struct, model=model)
