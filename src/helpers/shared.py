from typing import Dict, List, Literal, Optional, Tuple
from numpy.typing import NDArray

from depthai_sdk.components.camera_component import CameraComponent

class Sensor:
    component: Optional[CameraComponent] = None
    def __init__(
        self,
        socket: Literal["cam_a", "cam_b", "cam_c"],
        sensor_type: Literal["color", "depth"],
        width: int,
        height: int,
        frame_rate: int,
        color_order: Literal["rgb", "bgr"] = "rgb",
        interleaved: bool = False,
    ):
        self.socket = socket
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
            self._mapping[sensor.socket] = sensor

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
    """

    def __init__(self, np_array: NDArray, captured_at: float) -> None:
        self.np_array = np_array
        self.captured_at = captured_at
