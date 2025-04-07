import pytest

from viam.errors import ValidationError

from src.components.oak import Oak
from tests.helpers import make_component_config


sensors_not_present = (
    make_component_config(dict(), "viam:luxonis:oak-d"),
    'a "sensors" attribute of a list of sensor(s)'
)

sensors_is_not_list = (
    make_component_config({
        "sensors": "color"
    }, "viam:luxonis:oak-d"),
    "attribute must be a list_value"
)

sensors_is_empty_list = (
    make_component_config({
        "sensors": []
    }, "viam:luxonis:oak-d"),
    "attribute list cannot be empty"
)

sensors_list_too_long = (
    make_component_config({
        "sensors": ["color", "depth", "fake"]
    }, "viam:luxonis:oak-d"),
    "attribute list exceeds max length of two."
)

sensors_has_unknown_type = (
    make_component_config({
        "sensors": ["fake"]
    }, "viam:luxonis:oak-d"),
    "unknown sensor type"
)

sensors_has_duplicates = (
    make_component_config({
        "sensors": ["color", "color"]
    }, "viam:luxonis:oak-d"),
    "duplicates"
)

frame_rate_not_number_value = (
    make_component_config({
        "sensors": ["color", "depth"],
        "frame_rate": "30"
    }, "viam:luxonis:oak-d"),
    "attribute must be a number_value"
)

frame_rate_is_zero = (
    make_component_config({
        "sensors": ["color", "depth"],
        "frame_rate": 0
    }, "viam:luxonis:oak-d"),
    f"must be a float > 0"
)

frame_rate_is_negative = (
    make_component_config({
        "sensors": ["color", "depth"],
        "frame_rate": -1
    }, "viam:luxonis:oak-d"),
    f"must be a float > 0"
)

exposure_time_us_invalid = (
    make_component_config({
        "sensors": ["color", "depth"],
        "exposure_time_us": 100000,
        "iso": 100,
    }, "viam:luxonis:oak-d"),
    f"must be a integer between 1 and 33000"
)

iso_invalid = (
    make_component_config({
        "sensors": ["color", "depth"],
        "exposure_time_us": 1000,
        "iso": 10,
    }, "viam:luxonis:oak-d"),
    f"must be a integer between 100 and 1600"
)

iso_missing_exposure_time_us = (
    make_component_config({
        "sensors": ["color", "depth"],
        "iso": 100,
    }, "viam:luxonis:oak-d"),
    f"must be specified together"
)

dimension_not_number_value = (
    make_component_config({
        "sensors": ["color", "depth"],
        "height_px": "500px"
    }, "viam:luxonis:oak-d"),
    "attribute must be a number_value"
)

dimension_not_whole_number = (
    make_component_config({
        "sensors": ["color", "depth"],
        "height_px": 88.8
    }, "viam:luxonis:oak-d"),
    "must be a whole number"
)

height_is_zero = (
    make_component_config({
        "sensors": ["color", "depth"],
        "height_px": 0
    }, "viam:luxonis:oak-d"),
    "cannot be less than or equal to 0"
)

height_is_negative = (
    make_component_config({
        "sensors": ["color", "depth"],
        "height_px": -1
    }, "viam:luxonis:oak-d"),
    "cannot be less than or equal to 0"
)

width_is_zero = (
    make_component_config({
        "sensors": ["color", "depth"],
        "width_px": 0
    }, "viam:luxonis:oak-d"),
    "cannot be less than or equal to 0"
)

width_is_negative = (
    make_component_config({
        "sensors": ["color", "depth"],
        "width_px": -1
    }, "viam:luxonis:oak-d"),
    "cannot be less than or equal to 0"
)

only_received_height = (
    make_component_config({
        "sensors": ["color", "depth"],
        "height_px": 720
    }, "viam:luxonis:oak-d"),
    "received only one dimension attribute"
)

only_received_width = (
    make_component_config({
        "sensors": ["color", "depth"],
        "width_px": 1280
    }, "viam:luxonis:oak-d"),
    "received only one dimension attribute"
)

wrong_device_info_type = (
    make_component_config({
        "sensors": ["color", "depth"],
        "width_px": 1280,
        "height_px": 720,
        "device_info": 88
    }, "viam:luxonis:oak-d"),
    "attribute must be a string_value"
)

manual_focus_set_for_non_color = (
    make_component_config({
        "sensors": ["depth"],
        "manual_focus": 255
    }, "viam:luxonis:oak-d"),
    '"manual_focus" can be set only for the color sensor'
)

manual_focus_out_of_range = (
    make_component_config({
        "sensors": ["color", "depth"],
        "manual_focus": 256
    }, "viam:luxonis:oak-d"),
    '"manual_focus" must be a value in range 0...255 inclusive'
)

manual_focus_not_integer = (
    make_component_config({
        "sensors": ["color"],
        "manual_focus": 1.5
    }, "viam:luxonis:oak-d"),
    '"manual_focus" must be an integer'
)

right_handed_system_not_bool = (
    make_component_config({
        "sensors": ["color", "depth"],
        "right_handed_system": "true"
    }, "viam:luxonis:oak-d"),
    "attribute must be a bool_value"
)

point_cloud_enabled_not_bool = (
    make_component_config({
        "sensors": ["color", "depth"],
        "point_cloud_enabled": "true"
    }, "viam:luxonis:oak-d"),
    "attribute must be a bool_value"
)

configs_and_msgs = [
    sensors_not_present,
    sensors_is_not_list,
    sensors_is_empty_list,
    sensors_list_too_long,
    sensors_has_unknown_type,
    sensors_has_duplicates,
    frame_rate_not_number_value,
    frame_rate_is_zero,
    frame_rate_is_negative,
    exposure_time_us_invalid,
    iso_invalid,
    iso_missing_exposure_time_us,
    dimension_not_number_value,
    dimension_not_whole_number,
    height_is_zero,
    height_is_negative,
    width_is_zero,
    width_is_negative,
    only_received_height,
    only_received_width,
    wrong_device_info_type,
    manual_focus_set_for_non_color,
    manual_focus_out_of_range,
    manual_focus_not_integer,
    right_handed_system_not_bool,
    point_cloud_enabled_not_bool
]

full_correct_config = make_component_config({
    "sensors": ["color", "depth"],
    "height_px": 800,
    "width_px": 1280,
    "frame_rate": 60,
    "device_info": "18443010016B060F00"
}, "viam:luxonis:oak-d")

minimal_correct_config = make_component_config({
    "sensors": ["color"]
}, "viam:luxonis:oak-d")

@pytest.mark.parametrize("config,msg", configs_and_msgs)
def test_validate_errors_parameterized(config, msg):
    with pytest.raises(ValidationError) as exc_info:
        Oak.validate(config)
    assert exc_info.type == ValidationError
    assert msg in str(exc_info.value)

def test_validate_no_errors():
    try:
        Oak.validate(full_correct_config)
        Oak.validate(minimal_correct_config)
    except Exception as e:
        s = (f"Expected a correct config to not raise {type(e)} during validation, yet it did: {e}")
        pytest.fail(reason=s)
