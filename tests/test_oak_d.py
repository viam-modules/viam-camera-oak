import pytest

from viam.errors import ValidationError

from src.oak import Oak
from tests.test_helpers import make_component_config


invalid_attribute_name = (
    make_component_config({
        "sensors": ["color"],
        "foo": "bar"
    }, "viam:luxonis:oak-d"),
    "is not a valid attribute"
)

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

configs_and_msgs = [
    invalid_attribute_name,
    sensors_not_present,
    sensors_is_not_list,
    sensors_is_empty_list,
    sensors_list_too_long,
    sensors_has_unknown_type,
    sensors_has_duplicates,
    frame_rate_not_number_value,
    frame_rate_is_zero,
    frame_rate_is_negative,
    dimension_not_number_value,
    dimension_not_whole_number,
    height_is_zero,
    height_is_negative,
    width_is_zero,
    width_is_negative,
    only_received_height,
    only_received_width
]

full_correct_config = make_component_config({
    "sensors": ["color", "depth"],
    "height_px": 800,
    "width_px": 1280,
    "frame_rate": 60,
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
    except Exception as e:
        s = (f"Expected a correct config to not raise {type(e)} during validation, yet it did :,)")
        pytest.fail(reason=s)
