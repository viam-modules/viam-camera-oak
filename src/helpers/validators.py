from logging import Logger
from typing import Literal, Mapping, Optional

from google.protobuf.struct_pb2 import Struct

from viam.errors import ValidationError
from viam.proto.app.robot import ComponentConfig


from src.worker.worker import Worker


class Validator:
    def __init__(self, worker: Worker, config: ComponentConfig, logger: Logger):
        self.worker = worker
        self.config = config
        self.attribute_map = config.attributes.fields
        self.logger = logger

    def handle_err(self, err_msg: str) -> None:
        """
        handle_error is invoked when there is an error in validation.
        It logs a helpful error log, stops the worker if active, and
        raises & propagates the error
        """
        full_err_msg = f"Config attribute validation error: {err_msg}"
        self.logger.error(full_err_msg)
        if self.worker:  # stop worker if active
            self.worker.stop()
        raise ValidationError(full_err_msg)

    def validate_attr_type(
        self,
        attribute: str,
        expected_type: Literal[
            "null_value",
            "number_value",
            "string_value",
            "bool_value",
            "struct_value",
            "list_value",
        ],
        attribute_map: Optional[Mapping] = None,
        is_required_attr: Optional[bool] = False
    ) -> None:
        """
        Handles the existence of required/unrequired attributes. If it does exist, handles
        validating its type.
        """
        # can't set default in params to a self field, so set here
        if not attribute_map:
            attribute_map = self.attribute_map
        value = attribute_map.get(key=attribute, default=None)
        if value is None:
            if is_required_attr:
                self.handle_err(f'"{attribute}" is a required field. Please see README for details.')
            else:
                return
        if value.WhichOneof("kind") != expected_type:
            self.handle_err(
                f'the "{attribute}" attribute must be a {expected_type}, not {value}.'
            )

    def validate_dimension(self, attribute: str, attribute_map: Optional[Mapping] = None) -> None:
        """
        validate_dimension helps validates height_px and width_px values.
        """
        if attribute_map is None:
            attribute_map = self.attribute_map

        value = attribute_map.get(key=attribute, default=None)
        if value is None:
            return

        self.validate_attr_type(attribute, "number_value")
        number_value = value.number_value
        int_value = int(number_value)
        if int_value != number_value:
            self.handle_err(f'"{attribute}" must be a whole number.')
        if int_value <= 0:
            self.handle_err(
                f'inputted "{attribute}" of {int_value} cannot be less than or equal to 0.'
            )

    def validate_oak_d(self) -> None:
        """
        A procedure called to validate the OAK-D config.

        Args:
            config (ComponentConfig)

        Raises:
            ValidationError: with a description of what is wrong about the config

        Returns:
            None
        """
        VALID_ATTRIBUTES = ["height_px", "width_px", "sensors", "frame_rate"]
        # Check config keys are valid
        for attribute in self.attribute_map.keys():
            if attribute not in VALID_ATTRIBUTES:
                self.handle_err(
                    f'"{attribute}" is not a valid attribute i.e. {VALID_ATTRIBUTES}'
                )

        # Check sensors is valid
        sensors_value = self.attribute_map.get(key="sensors", default=None)
        if sensors_value is None:
            self.handle_err(
                """
                a "sensors" attribute of a list of sensor(s) is a required attribute e.g. ["depth", "color"],
                with the first sensor in the list being the main sensor that get_image uses.
                """
            )
        self.validate_attr_type("sensors", "list_value")
        sensor_list = list(sensors_value.list_value)
        if len(sensor_list) == 0:
            self.handle_err('"sensors" attribute list cannot be empty.')
        if len(sensor_list) > 2:
            self.handle_err('"sensors" attribute list exceeds max length of two.')
        for sensor in sensor_list:
            if sensor != "color" and sensor != "depth":
                self.handle_err(
                    f"""
                            unknown sensor type "{sensor}" found in "sensors" attribute list.
                            Valid sensors include: "color" and "depth"
                            """
                )
        if len(set(sensor_list)) != len(sensor_list):
            self.handle_err(
                f'please remove duplicates in the "sensors" attribute list: {sensor_list}'
            )

        # Validate frame rate
        self.validate_attr_type("frame_rate", "number_value")
        frame_rate = self.attribute_map.get(key="frame_rate", default=None)
        if frame_rate:
            if frame_rate.number_value <= 0:
                self.handle_err(f'"frame_rate" must be a float > 0.')

        # Check height_px value
        self.validate_dimension("height_px")

        # Check width_px value
        self.validate_dimension("width_px")

        # Check height_px and width_px together
        height_px, width_px = self.attribute_map.get(
            key="height_px", default=None
        ), self.attribute_map.get(key="width_px", default=None)
        if (height_px is None and width_px is not None) or (
            height_px is not None and width_px is None
        ):
            self.handle_err(
                'received only one dimension attribute. Please supply both "height_px" and "width_px", or neither.'
            )

    def validate_oak_ffc_3p(self) -> None:
        """
        A procedure called to validate the OAK-FFC-3P config.

        Args:
            config (ComponentConfig)

        Raises:
            ValidationError: with a description of what is wrong about the config

        Returns:
            None
        """
        # Validate outermost keys
        for k in self.attribute_map.keys():
            if k != "camera_sensors":
                self.handle_err(f'unrecognized attribute "{k}". Please fix or remove.')
        self.validate_attr_type("camera_sensors", "list_value", is_required_attr=True)
        cam_sensors_list = self.attribute_map.get("camera_sensors").list_value

        # Validate there are maximum 3 sensors and minimum 1
        if len(cam_sensors_list) == 0:
            self.handle_err('"camera_sensors" list cannot be empty.')
        elif len(cam_sensors_list) > 3:
            self.handle_err(
                '"camera_sensors" list cannot have >3 elements for the OAK-FFC-3P model.'
            )

        # Validate "type" attr. Check that there are either 0 or 2 depth sensors overall
        depth_sensor_count = 0
        for cam_sensor in cam_sensors_list:
            try: 
                cam_sensor.fields
            except AttributeError:
                self.handle_err('each cam_sensor must be a Struct mapping')

            self.validate_attr_type("type", "string_value", cam_sensor.fields, True)
            sensor_type = cam_sensor.fields.get(key="type", default=None).string_value
            if sensor_type not in ["color", "depth"]:
                self.handle_err(
                    f'the camera_sensor "type" attribute must be "color" or "depth". You provided: "{sensor_type}"'
                )
            if sensor_type == "depth":
                depth_sensor_count += 1
        if depth_sensor_count == 1 or depth_sensor_count == 3:
            self.handle_err(f"the OAK module supports 2 mono depth sensors at a time. You provided {depth_sensor_count}.")

        # Validate "socket"
        seen_sockets = []
        for cam_sensor in cam_sensors_list:
            self.validate_attr_type("socket", "string_value", cam_sensor.fields, True)
            socket = cam_sensor.fields.get("socket", default=None).string_value
            if socket not in ["cam_a", "cam_b", "cam_c"]:
                self.handle_err(f'"socket" attribute must be either "cam_a", "cam_b", or "cam_c", not "{socket}"')
            if socket in seen_sockets:
                self.handle_err(f'two or more camera_sensors were specified for socket {socket}. Please only specify 1.')
            seen_sockets.append(socket)

        # One more loop to validate rest of fields
        for cam_sensor in cam_sensors_list:
            # Validate "width_px", "height_px"
            self.validate_attr_type("width_px", "number_value", cam_sensor.fields, True)
            self.validate_attr_type("height_px", "number_value", cam_sensor.fields, True)
            self.validate_dimension("width_px", cam_sensor.fields)
            self.validate_dimension("height_px", cam_sensor.fields)

            # Validate "frame_rate"
            self.validate_attr_type("frame_rate", "number_value", cam_sensor.fields)
            frame_rate = cam_sensor.fields.get("frame_rate", None)
            if frame_rate and frame_rate.number_value <= 0:
                self.handle_err('"frame_rate" must be a float > 0.')

            # Validate "color_order"
            self.validate_attr_type("color_order", "string_value", cam_sensor.fields)
            color_order = cam_sensor.fields.get("color_order", None)
            if color_order and color_order.string_value not in ["rgb", "bgr"]:
                self.handle_err(f'"color_order" must be "rgb" or "bgr". You provided: "{color_order.string_value}"')

            # Validate "interleaved"
            self.validate_attr_type("interleaved", "bool_value", cam_sensor.fields)
