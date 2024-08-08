from typing import Dict, List, Literal, Optional, Tuple
from numpy.typing import NDArray

from depthai import CameraBoardSocket


def get_socket_from_str(s: str) -> CameraBoardSocket:
    if s == "cam_a":
        return CameraBoardSocket.CAM_A
    elif s == "cam_b":
        return CameraBoardSocket.CAM_B
    elif s == "cam_c":
        return CameraBoardSocket.CAM_C
    else:
        raise Exception(f"Camera socket '{s}' is not recognized or supported.")


class Sensor:
    def get_unique_name(self) -> str:
        if self.sensor_type == "color":
            return f"{self.socket_str}_rgb"
        else:
            return f"{self.socket_str}_mono"

    def __init__(
        self,
        socket_str: Literal["cam_a", "cam_b", "cam_c"],
        sensor_type: Literal["color", "depth"],
        width: int,
        height: int,
        frame_rate: int,
        color_order: Literal["rgb", "bgr"] = "rgb",
        interleaved: bool = False,
    ):
        self.socket_str = socket_str
        self.socket = get_socket_from_str(socket_str)
        self.sensor_type = sensor_type
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.color_order = color_order
        self.interleaved = interleaved


class Sensors:
    _mapping: Dict[str, Sensor]
    stereo_pair: Optional[Tuple[Sensor, Sensor]]
    color_sensors: Optional[List[Sensor]]
    primary_sensor: Sensor

    def __init__(self, sensors: List[Sensor]):
        self._mapping = dict()
        for sensor in sensors:
            self._mapping[sensor.socket_str] = sensor

        self.color_sensors = self._find_color_sensors()
        self.stereo_pair = self._find_stereo_pair()
        self.primary_sensor = sensors[0]

    def get_cam_a(self) -> Sensor:
        return self._mapping["cam_a"]

    def get_cam_b(self) -> Sensor:
        return self._mapping["cam_b"]

    def get_cam_c(self) -> Sensor:
        return self._mapping["cam_c"]

    def _find_color_sensors(self) -> List[Sensor]:
        l = []
        for sensor in self._mapping.values():
            if sensor.sensor_type == "color":
                l.append(sensor)
        return l

    def _find_stereo_pair(self) -> Optional[Tuple[Sensor]]:
        pair = []
        for sensor in self._mapping.values():
            if sensor.sensor_type == "depth":
                pair.append(sensor)
        if len(pair) == 0:
            return None
        return tuple(pair)


class CapturedData:
    """
    CapturedData is image data with the data as an np array,
    plus the timestamp it was captured at.

    This class uses __slots__ to:
        - Optimize memory usage by pre-defining attributes.
        - Speed up attribute access slightly.
        - Prevent unidiomatic additions of new attributes at runtime.
    """

    __slots__ = ("np_array", "captured_at")

    def __init__(self, np_array: NDArray, captured_at: float) -> None:
        self.np_array = np_array
        self.captured_at = captured_at
