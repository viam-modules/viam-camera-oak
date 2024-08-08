from typing import List, Literal, Mapping, Optional

from google.protobuf.struct_pb2 import Value

from viam.errors import ValidationError
from viam.logging import getLogger
from src.components.helpers.shared import Sensor, Sensors


# Be sure to update README.md if default attributes are changed
DEFAULT_FRAME_RATE = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_COLOR_ORDER = "rgb"
DEFAULT_INTERLEAVED = False
LOGGER = getLogger("viam-oak-config-logger")


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
                f'"{attribute}" is a required field, but was not detected. Please see README for details.'
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
    def __init__(self, attribute_map: Mapping[str, Value]):
        self.attribute_map = attribute_map

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value]) -> List[str]:
        """
        Equivalent to the validate() method for modules.

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
    device_info: str
    sensors: Sensors

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value]) -> List[str]:
        # Validate shared OAK attributes such as "device_info"
        validate_attr_type("device_info", "string_value", attribute_map)
        device_info = attribute_map.get(key="device_info", default=None)
        if device_info is None:
            LOGGER.info(
                '"device_info" attribute unspecified. Will default to the first OAK device detected.'
            )


class OakDConfig(OakConfig):
    def initialize_config(self):
        self.device_info = self.attribute_map["device_info"].string_value or None
        sensors_str_list = list(self.attribute_map["sensors"].list_value)

        height = int(self.attribute_map["height_px"].number_value) or DEFAULT_HEIGHT
        width = int(self.attribute_map["width_px"].number_value) or DEFAULT_WIDTH
        frame_rate = self.attribute_map["frame_rate"].number_value or DEFAULT_FRAME_RATE

        sensor_list = []
        for sensor_str in sensors_str_list:
            if sensor_str == "depth":
                for cam_socket in ["cam_b", "cam_c"]:
                    depth_sensor = Sensor(
                        cam_socket, "depth", width, height, frame_rate
                    )
                    sensor_list.append(depth_sensor)
            elif sensor_str == "color":
                color_sensor = Sensor(
                    "cam_a", "color", width, height, frame_rate, "rgb"
                )
                sensor_list.append(color_sensor)
        self.sensors = Sensors(sensor_list)

    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value]) -> List[str]:
        super().validate(attribute_map)

        VALID_ATTRIBUTES = [
            "height_px",
            "width_px",
            "sensors",
            "frame_rate",
            "device_info",
        ]
        # Check config keys are valid
        for attribute in attribute_map.keys():
            if attribute not in VALID_ATTRIBUTES:
                handle_err(
                    f'"{attribute}" is not a valid attribute i.e. {VALID_ATTRIBUTES}'
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

        return []  # no deps


class OakFfc3PConfig(OakConfig):
    @classmethod
    def validate(cls, attribute_map: Mapping[str, Value]) -> List[str]:
        super().validate(attribute_map)

        # Validate outermost keys
        for k in attribute_map.keys():
            if k != "camera_sensors":
                handle_err(f'unrecognized attribute "{k}". Please fix or remove.')
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
    # Default values for non-required attributes are set here
    cam_name: str
    input_source: str
    width: int
    height: int
    num_threads: int = 1
    num_nce_per_thread: int = 1
    is_object_tracker: bool = False

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
        self = cls(dict())
        self.input_source = kwargs["input_source"]
        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.num_threads = kwargs.get("num_threads", self.num_threads)
        self.num_nce_per_thread = kwargs.get(
            "num_nce_per_thread", self.num_nce_per_thread
        )
        self.is_object_tracker = kwargs.get("is_object_tracker", self.is_object_tracker)

        self.blob_path = kwargs["blob_path"]
        self.labels = kwargs["labels"]
        self.confidence_threshold = kwargs.get(
            "confidence_threshold", self.confidence_threshold
        )
        self.iou_threshold = kwargs.get("iou_threshold", self.iou_threshold)
        self.anchors = kwargs.get("anchors", [])
        self.anchor_masks = kwargs.get("anchor_masks", dict())
        for l in self.anchor_masks.values():
            for i, num in enumerate(l):
                l[i] = int(num)
        self.coordinate_size = kwargs.get("coordinate_size", self.coordinate_size)

        self.service_name = kwargs["service_name"]
        self.service_id = kwargs["service_id"]
        return self

    @classmethod
    def validate(self, attribute_map: Mapping[str, Value]) -> List[str]:
        # Validate "input_source"
        validate_attr_type("input_source", "string_value", attribute_map, True)
        input_source = attribute_map.get("input_source", default=None).string_value
        if input_source not in ["cam_a", "cam_b", "cam_c", "color"]:
            handle_err(
                f'"input_source" attribute must be either "color", "cam_a", "cam_b", or "cam_c", not "{input_source}"'
            )

        # Validate "width_px" and "height_px"
        validate_attr_type("width_px", "number_value", attribute_map, True)
        validate_attr_type("height_px", "number_value", attribute_map, True)
        validate_dimension("width_px", attribute_map)
        validate_dimension("height_px", attribute_map)

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

        # Validate "is_object_tracker"
        validate_attr_type("is_object_tracker", "bool_value", attribute_map)

        # Validate "yolo_cfg" and those nested fields
        validate_attr_type("yolo_config", "struct_value", attribute_map, True)
        yolo_cfg = attribute_map.get("yolo_config").struct_value.fields

        # Validate "blob_path"
        validate_attr_type("blob_path", "string_value", yolo_cfg, True)
        if len(yolo_cfg.get("blob_path").string_value) == 0:
            handle_err('"blob_path" cannot be empty string.')

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
        self.width = int(self.attribute_map["width_px"].number_value)
        self.height = int(self.attribute_map["height_px"].number_value)
        self.num_threads = (
            int(self.attribute_map["num_threads"].number_value) or self.num_threads
        )
        self.num_nce_per_thread = (
            int(self.attribute_map["num_nce_per_thread"].number_value)
            or self.num_nce_per_thread
        )
        self.is_object_tracker = (
            self.attribute_map["is_object_tracker"].bool_value or self.is_object_tracker
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
