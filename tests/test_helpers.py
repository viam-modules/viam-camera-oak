from typing import Any, Mapping
from google.protobuf.struct_pb2 import Struct

from viam.proto.app.robot import ComponentConfig

def make_component_config(dictionary: Mapping[str, Any], model: str) -> ComponentConfig:
    struct = Struct()
    struct.update(dictionary=dictionary)
    return ComponentConfig(attributes=struct, model=model)
