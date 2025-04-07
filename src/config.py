import os
from logging import Logger
from typing import ClassVar, Dict, List, Literal, Mapping, Optional, Tuple

from google.protobuf.struct_pb2 import Value

from viam.errors import ValidationError
from viam.logging import getLogger
from src.components.helpers.shared import get_socket_from_str


# Be sure to update README.md if default attributes are changed
DEFAULT_FRAME_RATE = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_COLOR_ORDER = "rgb"
DEFAULT_INTERLEAVED = False
LOGGER = getLogger("viam-luxonis-configuration")


class Sensor:
    """
    Sensor config. Corresponds to a socket and what camera should be configured
    off of the specified socket.
    """

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
        manual_focus: Optional[int] = None,
    ):
        self.socket_str = socket_str
        self.socket = get_socket_from_str(socket_str)
        self.sensor_type = sensor_type
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.color_order = color_order
        self.interleaved = interleaved
        self.manual_focus = manual_focus


class Sensors:
    """
    Sensors wraps a Sensor mapping and offers handy utility methods and fields.
    """

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


def handle_err(err_msg: str) -> None:
    """
    handle_error is invoked when there is an error in validation.
    It logs a helpful error log, stops the worker if active, and
    raises & propagates the error
    """
    full_err_msg = f"Config attribute validation error: {err_msg}"
    LOGGER.error(full_err_msg)
    raise ValidationError(full_err_msg)


def validate_attr_type(
    attribute: str,
    expected_type: Literal[
        "null_value",
        "number_value",
        "string_value",
        "bool_value",
        "struct_value",
        "list_value",
    ],
    attribute_map: Mapping[str, Value],
    is_required_attr: Optional[bool] = False,
) -> None:
    """
    Handles the existence of required/unrequired attributes. If it does exist, handles
    validating its type.
    """
    value = attribute_map.get(key=attribute, default=None)
    if value is None:
        if is_required_attr:
            handle_err(
                f'"{attribute}" is a required field, but was not detected. Please see module docs in app configuration card.'
            )
        else:
            return
    if value.WhichOneof("kind") != expected_type:
        handle_err(
            f'the "{attribute}" attribute must be a {expected_type}, not {value}.'
        )


def validate_dimension(attribute: str, attribute_map: Mapping[str, Value]) -> None:
    """
    validate_dimension helps validates height_px and width_px values.
    """
    value = attribute_map.get(key=attribute, default=None)
    if value is None:
        return

    validate_attr_type(attribute, "number_value", attribute_map)
    number_value = value.number_value
    int_value = int(number_value)
    if int_value != number_value:
        handle_err(f'"{attribute}" must be a whole number.')
    if int_value <= 0:
        handle_err(
            f'inputted "{attribute}" of {int_value} cannot be less than or equal to 0.'
        )


class BaseConfig:
    """
    Base class for native configurations for all models in this module.
    """

    def __init__(self, attribute_map: Mapping[str, Value], name: str):
        self.attribute_map = attribute_map
        self.name = name

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value], logger: Logger) -> List[str]:
        """
        Equivalent to the module validate() method but specific to a model's specific config.

        Subclasses should inherit and implement this method.

        Returns:
            List[str]: deps if any
        """
        pass

    def initialize_config(self) -> None:
        """
        Takes the input config and turns it into the native config stored as attributes.

        Subclasses should inherit and implement this method.

        Returns:
            None
        """
        pass


class OakConfig(BaseConfig):
    """
    Base config class for OAK component models.
    """

    device_info: Optional[str]
    sensors: Sensors

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value], logger: Logger) -> List[str]:
        # Validate shared OAK attributes such as "device_info"
        validate_attr_type("device_info", "string_value", attribute_map)
        device_info = attribute_map.get(key="device_info", default=None)
        if device_info is None:
            LOGGER.info(
                '"device_info" attribute unspecified. Will default to the first OAK device detected.'
            )


class OakDConfig(OakConfig):
    """
    OAK-D component model native config
    """

    VALID_ATTRIBUTES: ClassVar[List[str]] = [
        "device_info",
        "sensors",
        "height_px",
        "width_px",
        "frame_rate",
        "exposure_time_us",
        "iso",
        "manual_focus",
        "right_handed_system",
        "point_cloud_enabled",
    ]

    height_px: int
    width_px: int
    frame_rate: int
    exposure_time_us: Optional[int]
    iso: Optional[int]
    manual_focus: Optional[int]
    right_handed_system: bool

    def initialize_config(self):
        self.device_info = self.attribute_map["device_info"].string_value or None
        sensors_str_list = list(self.attribute_map["sensors"].list_value)

        height = int(self.attribute_map["height_px"].number_value) or DEFAULT_HEIGHT
        width = int(self.attribute_map["width_px"].number_value) or DEFAULT_WIDTH
        frame_rate = self.attribute_map["frame_rate"].number_value or DEFAULT_FRAME_RATE
        if self.attribute_map["exposure_time_us"].number_value:
            self.exposure_time_us = self.attribute_map["exposure_time_us"].number_value
        if self.attribute_map["iso"].number_value:
            self.iso = self.attribute_map["iso"].number_value

        manual_focus = int(self.attribute_map["manual_focus"].number_value) or None
        self.right_handed_system = (
            self.attribute_map["right_handed_system"].bool_value or False
        )
        self.point_cloud_enabled = (
            self.attribute_map["point_cloud_enabled"].bool_value or False
        )

        sensor_list = []
        for sensor_str in sensors_str_list:
            if sensor_str == "depth":
                for cam_socket in ["cam_b", "cam_c"]:
                    depth_sensor = Sensor(
                        socket_str=cam_socket,
                        sensor_type="depth",
                        width=width,
                        height=height,
                        frame_rate=frame_rate,
                    )
                    sensor_list.append(depth_sensor)
            elif sensor_str == "color":
                color_sensor = Sensor(
                    socket_str="cam_a",
                    sensor_type="color",
                    width=width,
                    height=height,
                    frame_rate=frame_rate,
                    color_order="rgb",
                    manual_focus=manual_focus,
                )
                sensor_list.append(color_sensor)
        self.sensors = Sensors(sensor_list)

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value], logger: Logger) -> List[str]:
        super().validate(attribute_map, logger)

        # Validate outermost keys
        for k in attribute_map.keys():
            if k not in cls.VALID_ATTRIBUTES:
                logger.warning(
                    f'"{k}" is not a valid attribute i.e. not in {cls.VALID_ATTRIBUTES}. Please see module docs in app configuration card.'
                )

        # Check sensors is valid
        sensors_value = attribute_map.get(key="sensors", default=None)
        if sensors_value is None:
            handle_err(
                """
                a "sensors" attribute of a list of sensor(s) is a required attribute e.g. ["depth", "color"],
                with the first sensor in the list being the main sensor that get_image uses.
                """
            )
        validate_attr_type("sensors", "list_value", attribute_map)
        sensor_list = list(sensors_value.list_value)
        if len(sensor_list) == 0:
            handle_err('"sensors" attribute list cannot be empty.')
        if len(sensor_list) > 2:
            handle_err('"sensors" attribute list exceeds max length of two.')
        for sensor in sensor_list:
            if sensor != "color" and sensor != "depth":
                handle_err(
                    f"""
                            unknown sensor type "{sensor}" found in "sensors" attribute list.
                            Valid sensors include: "color" and "depth"
                            """
                )
        if len(set(sensor_list)) != len(sensor_list):
            handle_err(
                f'please remove duplicates in the "sensors" attribute list: {sensor_list}'
            )

        # Validate frame rate
        validate_attr_type("frame_rate", "number_value", attribute_map)
        frame_rate = attribute_map.get(key="frame_rate", default=None)
        if frame_rate:
            if frame_rate.number_value <= 0:
                handle_err(f'"frame_rate" must be a float > 0.')

        # Validate exposure_time_us
        validate_attr_type("exposure_time_us", "number_value", attribute_map)
        exposure_time_us = attribute_map.get(key="exposure_time_us", default=None)
        if exposure_time_us:
            if (exposure_time_us.number_value > 33000) or (
                exposure_time_us.number_value <= 0
            ):
                handle_err(
                    f'"exposure_time_us" must be a integer between 1 and 33000 inclusive.'
                )

        # Validate iso
        validate_attr_type("iso", "number_value", attribute_map)
        iso = attribute_map.get(key="iso", default=None)
        if iso:
            if (iso.number_value > 1600) or (iso.number_value < 100):
                handle_err(f'"iso" must be a integer between 100 and 1600 inclusive.')

        if (exposure_time_us and not iso) or (iso and not exposure_time_us):
            handle_err(f'"exposure_time_us" and "iso" must be specified together')

        # Check height_px value
        validate_dimension("height_px", attribute_map)

        # Check width_px value
        validate_dimension("width_px", attribute_map)

        # Check height_px and width_px together
        height_px, width_px = attribute_map.get(
            key="height_px", default=None
        ), attribute_map.get(key="width_px", default=None)
        if (height_px is None and width_px is not None) or (
            height_px is not None and width_px is None
        ):
            handle_err(
                'received only one dimension attribute. Please supply both "height_px" and "width_px", or neither.'
            )

        # Validate manual_focus
        validate_attr_type("manual_focus", "number_value", attribute_map)
        manual_focus = attribute_map.get(key="manual_focus", default=None)
        if manual_focus:
            if "color" not in sensor_list:
                handle_err('"manual_focus" can be set only for the color sensor')

            focus_value = manual_focus.number_value
            if focus_value < 0 or focus_value > 255:
                handle_err('"manual_focus" must be a value in range 0...255 inclusive')
            if int(focus_value) != focus_value:
                handle_err('"manual_focus" must be an integer')

        # Validate right_handed_system
        validate_attr_type("right_handed_system", "bool_value", attribute_map, False)

        # Validate point_cloud_enabled
        validate_attr_type("point_cloud_enabled", "bool_value", attribute_map, False)
        return []  # no deps


class OakFfc3PConfig(OakConfig):
    """
    Native config for OAK-FFC-3P component model.
    """

    VALID_ATTRIBUTES: ClassVar[List[str]] = ["device_info", "camera_sensors"]

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value], logger: Logger) -> List[str]:
        super().validate(attribute_map, logger)

        # Validate outermost keys
        for k in attribute_map.keys():
            if k not in cls.VALID_ATTRIBUTES:
                logger.warning(
                    f'"{k}" is not a valid attribute i.e. not in {cls.VALID_ATTRIBUTES}. Please see module docs in app configuration card.'
                )

        # Validate "camera_sensors"
        validate_attr_type("camera_sensors", "list_value", attribute_map, True)
        cam_sensors_list = attribute_map.get("camera_sensors").list_value

        # Validate there are maximum 3 sensors and minimum 1
        if len(cam_sensors_list) == 0:
            handle_err('"camera_sensors" list cannot be empty.')
        elif len(cam_sensors_list) > 3:
            handle_err(
                '"camera_sensors" list cannot have >3 elements for the OAK-FFC-3P model.'
            )

        # Validate "type" attr. Check that there are either 0 or 2 depth sensors overall
        depth_sensor_count = 0
        for cam_sensor in cam_sensors_list:
            try:
                cam_sensor.fields
            except AttributeError:
                handle_err("each cam_sensor must be a Struct mapping")

            validate_attr_type("type", "string_value", cam_sensor.fields, True)
            sensor_type = cam_sensor.fields.get(key="type", default=None).string_value
            if sensor_type not in ["color", "depth"]:
                handle_err(
                    f'the camera_sensor "type" attribute must be "color" or "depth". You provided: "{sensor_type}"'
                )
            if sensor_type == "depth":
                depth_sensor_count += 1
        if depth_sensor_count == 1 or depth_sensor_count == 3:
            handle_err(
                f"the OAK module supports 2 mono depth sensors at a time. You provided {depth_sensor_count}."
            )

        # Validate "socket"
        seen_sockets = []
        for cam_sensor in cam_sensors_list:
            validate_attr_type("socket", "string_value", cam_sensor.fields, True)
            socket = cam_sensor.fields.get("socket", default=None).string_value
            if socket not in ["cam_a", "cam_b", "cam_c"]:
                handle_err(
                    f'"socket" attribute must be either "cam_a", "cam_b", or "cam_c", not "{socket}"'
                )
            if socket in seen_sockets:
                handle_err(
                    f"two or more camera_sensors were specified for socket {socket}. Please only specify 1."
                )
            seen_sockets.append(socket)

        # One more loop to validate rest of fields
        for cam_sensor in cam_sensors_list:
            # Validate "width_px", "height_px"
            validate_attr_type("width_px", "number_value", cam_sensor.fields, True)
            validate_attr_type("height_px", "number_value", cam_sensor.fields, True)
            validate_dimension("width_px", cam_sensor.fields)
            validate_dimension("height_px", cam_sensor.fields)

            # Validate "frame_rate"
            validate_attr_type("frame_rate", "number_value", cam_sensor.fields)
            frame_rate = cam_sensor.fields.get("frame_rate", None)
            if frame_rate and frame_rate.number_value <= 0:
                handle_err('"frame_rate" must be a float > 0.')

            # Validate "color_order"
            validate_attr_type("color_order", "string_value", cam_sensor.fields)
            color_order = cam_sensor.fields.get("color_order", None)
            if color_order and color_order.string_value not in ["rgb", "bgr"]:
                handle_err(
                    f'"color_order" must be "rgb" or "bgr". You provided: "{color_order.string_value}"'
                )

            # Validate "interleaved"
            validate_attr_type("interleaved", "bool_value", cam_sensor.fields)

        return []  # no deps

    def initialize_config(self):
        self.device_info = self.attribute_map["device_info"].string_value or None
        cam_sensors_list = self.attribute_map["camera_sensors"].list_value

        sensor_list = []
        for cam_sensor_struct in cam_sensors_list:
            fields = cam_sensor_struct.fields
            socket = fields.get("socket").string_value
            sensor_type = fields.get("type").string_value
            width = int(fields.get("width_px").number_value)
            height = int(fields.get("height_px").number_value)
            frame_rate = fields["frame_rate"].number_value or DEFAULT_FRAME_RATE
            color_order = fields["color_order"].string_value or DEFAULT_COLOR_ORDER
            interleaved = fields["interleaved"].bool_value or DEFAULT_INTERLEAVED

            sensor = Sensor(
                socket, sensor_type, width, height, frame_rate, color_order, interleaved
            )
            sensor_list.append(sensor)
        self.sensors = Sensors(sensor_list)


class YDNConfig(BaseConfig):
    """
    Native config for configuring a yolo detection network in the DepthAI pipeline.
    """

    VALID_ATTRIBUTES: ClassVar[List[str]] = [
        "cam_name",
        "input_source",
        "yolo_config",
        "num_nce_per_thread",
        "num_threads",
    ]

    # Default values for non-required attributes are set here
    cam_name: str
    input_source: str
    num_threads: int = 1
    num_nce_per_thread: int = 1

    blob_path: str
    labels: List[str]
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    anchors: List[float]
    anchor_masks: Mapping[str, List[int]]
    coordinate_size: int = 4

    # Used in OAK to check what service this config is for
    service_name: str
    service_id: str

    @classmethod
    def from_kwargs(cls, **kwargs):
        self = cls(dict(), kwargs["service_name"])
        self.input_source = kwargs["input_source"]
        self.num_threads = kwargs.get("num_threads", self.num_threads)
        self.num_nce_per_thread = kwargs.get(
            "num_nce_per_thread", self.num_nce_per_thread
        )

        self.blob_path = kwargs["blob_path"]
        self.labels = kwargs["labels"]
        self.confidence_threshold = kwargs.get(
            "confidence_threshold", self.confidence_threshold
        )
        self.iou_threshold = kwargs.get("iou_threshold", self.iou_threshold)
        self.anchors = kwargs.get("anchors", [])
        self.anchor_masks = kwargs.get("anchor_masks", dict())
        self.anchor_masks = {
            key: list(map(int, value))
            for key, value in kwargs.get("anchor_masks", {}).items()
        }
        self.coordinate_size = kwargs.get("coordinate_size", self.coordinate_size)

        self.service_name = kwargs["service_name"]
        self.service_id = kwargs["service_id"]
        return self

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value], logger: Logger) -> List[str]:
        super().validate(attribute_map, logger)

        # Validate outermost keys
        for k in attribute_map.keys():
            if k not in cls.VALID_ATTRIBUTES:
                logger.warning(
                    f'"{k}" is not a valid attribute i.e. not in {cls.VALID_ATTRIBUTES}. Please see module docs in app configuration card.'
                )

        # Validate "input_source"
        validate_attr_type("input_source", "string_value", attribute_map, True)
        input_source = attribute_map.get("input_source", default=None).string_value
        if input_source not in ["cam_a", "cam_b", "cam_c", "color"]:
            handle_err(
                f'"input_source" attribute must be either "color", "cam_a", "cam_b", or "cam_c", not "{input_source}"'
            )

        # Validate "num_threads"
        validate_attr_type("num_threads", "number_value", attribute_map)
        num_threads_container = attribute_map.get("num_threads", None)
        if num_threads_container is not None:
            num_threads = num_threads_container.number_value
            if num_threads not in [0, 1, 2]:
                handle_err(
                    f'"num_threads" must be 0, 1, or 2. 0 means AUTO. You set {num_threads}'
                )

        # Validate "num_nce_per_thread"
        validate_attr_type("num_nce_per_thread", "number_value", attribute_map)
        num_nce_container = attribute_map.get("num_nce_per_thread", None)
        if num_nce_container is not None:
            num_nce = num_nce_container.number_value
            if num_nce not in [1, 2]:
                handle_err(f'"num_nce_per_thread" must be 1 or 2. You set {num_nce}')

        # Validate "yolo_cfg" and those nested fields
        validate_attr_type("yolo_config", "struct_value", attribute_map, True)
        yolo_cfg = attribute_map.get("yolo_config").struct_value.fields

        # Validate "blob_path"
        validate_attr_type("blob_path", "string_value", yolo_cfg, True)
        blob_path_value = yolo_cfg.get("blob_path").string_value
        if len(blob_path_value) == 0:
            handle_err('"blob_path" cannot be empty string.')

        # Convert to absolute path if relative
        try:
            blob_path_value = os.path.expanduser(blob_path_value)
            blob_path_value = os.path.expandvars(blob_path_value)
            if not os.path.isabs(blob_path_value):
                blob_path_value = os.path.abspath(blob_path_value)
        except Exception as e:
            handle_err(f"Invalid blob_path: {blob_path_value}. Error: {str(e)}")
        # Update the blob_path in yolo config
        yolo_cfg["blob_path"].string_value = blob_path_value

        # Validate "labels"
        validate_attr_type("labels", "list_value", yolo_cfg, True)
        labels: List[Value] = yolo_cfg.get("labels").list_value
        if len(labels) == 0:
            handle_err('"labels" attribute list cannot be empty.')
        for label in labels:
            if not isinstance(label, str):
                handle_err('List elements of "labels" must be strings.')
            if len(label) == 0:
                handle_err('List elements of "labels" cannot be empty strings.')

        # Validate "confidence_threshold"
        validate_attr_type("confidence_threshold", "number_value", yolo_cfg)
        confidence_threshold_container = yolo_cfg.get("confidence_threshold", None)
        if confidence_threshold_container is not None:
            confidence_threshold = confidence_threshold_container.number_value
            if confidence_threshold > 1 or confidence_threshold < 0:
                handle_err(
                    f'"confidence_threshold" must be between 0 and 1. You set {confidence_threshold}'
                )

        # Validate "iou_threshold"
        validate_attr_type("iou_threshold", "number_value", yolo_cfg)
        iou_threshold_container = yolo_cfg.get("iou_threshold", None)
        if iou_threshold_container is not None:
            iou_threshold = iou_threshold_container.number_value
            if iou_threshold > 1 or iou_threshold < 0:
                handle_err(
                    f'"iou_threshold" must be between 0 and 1. You set {iou_threshold}'
                )

        # Validate "anchors"
        validate_attr_type("anchors", "list_value", yolo_cfg)
        if yolo_cfg.get("anchors", None) is not None:
            anchors: List[Value] = yolo_cfg.get("anchors").list_value
            if len(anchors) == 0:
                handle_err('"labels" attribute list cannot be empty.')
            for anchor in anchors:
                if not isinstance(anchor, float) or float(anchor) != int(anchor):
                    handle_err('List elements of "anchors" must be integers.')
                if anchor < 0:
                    handle_err('List elements of "anchors" cannot be negative.')

        # Validate "anchor_masks"
        validate_attr_type("anchor_masks", "struct_value", yolo_cfg)
        anchor_masks_container = yolo_cfg.get("anchor_masks", None)
        if anchor_masks_container is not None:
            anchor_mask_mapping = anchor_masks_container.struct_value.fields
            if len(anchor_mask_mapping) == 0:
                handle_err(
                    'No anchor masks found in "anchor_masks". Please either supply them or remove the attribute entirely.'
                )
            for field_name, anchors in anchor_mask_mapping.items():
                if anchors.WhichOneof("kind") != "list_value":
                    handle_err(
                        f'Anchor mask subfield "{field_name}" must be a list value.'
                    )
                anchor_list = anchors.list_value
                if len(anchor_list) == 0:
                    handle_err(
                        f'Anchor mask list subfield "{field_name}" cannot be empty.'
                    )
                for anchor in anchor_list:
                    if not isinstance(anchor, float) or float(anchor) != int(anchor):
                        handle_err('List elements of "anchors" must be integers.')
                    if anchor < 0:
                        handle_err('List elements of "anchors" cannot be negative.')

        # Validate "coordinate_size"
        validate_attr_type("coordinate_size", "number_value", yolo_cfg)
        coordinate_size_container = yolo_cfg.get("coordinate_size", None)
        if coordinate_size_container is not None:
            coordinate_size = coordinate_size_container.number_value
            if coordinate_size < 0:
                handle_err(
                    f'"coordinate_size" attribute must be positive, not {coordinate_size}'
                )

        # Validate "cam_name"
        validate_attr_type("cam_name", "string_value", attribute_map, True)
        cam_name_container = attribute_map.get("cam_name", None)
        if cam_name_container is None:
            handle_err(
                "Critical logic error: cam_name should not be None since we should've asserted it to not be. This is likely a bug."
            )
        cam_name = cam_name_container.string_value
        if len(cam_name) == 0:
            handle_err('"cam_name" attribute cannot be empty string.')
        return [cam_name]

    def initialize_config(self):
        self.cam_name = self.attribute_map["cam_name"].string_value
        self.input_source = self.attribute_map["input_source"].string_value
        self.num_threads = (
            int(self.attribute_map["num_threads"].number_value) or self.num_threads
        )
        self.num_nce_per_thread = (
            int(self.attribute_map["num_nce_per_thread"].number_value)
            or self.num_nce_per_thread
        )

        yolo_cfg = self.attribute_map["yolo_config"].struct_value.fields
        self.blob_path = yolo_cfg["blob_path"].string_value
        self.labels = yolo_cfg["labels"].list_value
        self.confidence_threshold = (
            yolo_cfg["confidence_threshold"].number_value or self.confidence_threshold
        )
        self.iou_threshold = (
            yolo_cfg["iou_threshold"].number_value or self.iou_threshold
        )
        self.anchors = yolo_cfg["anchors"].list_value if "anchors" in yolo_cfg else []
        self.anchor_masks = (
            {
                field_name: anchors.list_value
                for field_name, anchors in yolo_cfg[
                    "anchor_masks"
                ].struct_value.fields.items()
            }
            if "anchor_masks" in yolo_cfg
            else dict()
        )
        self.coordinate_size = (
            int(yolo_cfg["coordinate_size"].number_value) or self.coordinate_size
        )
