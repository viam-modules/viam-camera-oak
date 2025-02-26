import pytest

from viam.errors import ValidationError

from src.components.oak import Oak
from tests.helpers import make_component_config


### TEST INVALID CONFIGS

empty_sensors_config = (
    make_component_config({
        "camera_sensors": [],
    }, "viam:luxonis:oak-ffc-3p"),
    "list cannot be empty"
)

over_three_sensors_config = (
    make_component_config({
        "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            {
                "socket": "cam_b",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            {
                "socket": "cam_c",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            {
                "socket": "cam_d",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
    "list cannot have >3 elements for the OAK-FFC-3P model"
)

camera_sensors_wrong_type_config = (
    make_component_config({
        "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            "oopsies",
        ],
    }, "viam:luxonis:oak-ffc-3p"),
    "each cam_sensor must be a Struct mapping"
)

two_cams_one_socket_config = (
    make_component_config({
        "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            {
                "socket": "cam_a",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
    "two or more camera_sensors were specified for socket"
)

one_mono_config = (
    make_component_config({
        "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            }
        ],
    }, "viam:luxonis:oak-ffc-3p"),
    "the OAK module supports 2 mono depth sensors at a time. You provided 1"
)

three_mono_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            {
                "socket": "cam_b",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            {
                "socket": "cam_c",
                "type": "depth",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
    'the OAK module supports 2 mono depth sensors at a time. You provided 3'
)

invalid_sensor_type_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "mono",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'the camera_sensor "type" attribute must be "color" or "depth". You provided: "mono"'
)

invalid_socket_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_d",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   '"socket" attribute must be either "cam_a", "cam_b", or "cam_c", not "cam_d"'
)

conflicting_sockets_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'two or more camera_sensors were specified for socket cam_a. Please only specify 1.'
)

missing_width_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   '"width_px" is a required field, but was not detected. Please see module docs in app configuration card.'
)

missing_height_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   '"height_px" is a required field, but was not detected. Please see module docs in app configuration card.'
)

dimension_not_num_type_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": "512px",
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'attribute must be a number_value'
)

dimension_not_whole_num_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512.1,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'must be a whole number.'
)

dimension_is_zero_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 0,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'cannot be less than or equal to 0'
)

dimension_is_neg_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": -512,
                "frame_rate": 30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'cannot be less than or equal to 0'
)

frame_rate_not_num_type = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": "30",
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'attribute must be a number_value'
)

frame_rate_is_zero = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 0,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   '"frame_rate" must be a float > 0.'
)

frame_rate_is_neg = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": -30,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   '"frame_rate" must be a float > 0.'
)

non_string_color_order_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "color_order": False,
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'attribute must be a string_value'
)

wrong_string_color_order_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "color_order": "rbg",
                "interleaved": False,
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   '"color_order" must be "rgb" or "bgr". You provided: "rbg"'
)

non_bool_interleaved_config = (
    make_component_config({
            "camera_sensors": [
            {
                "socket": "cam_a",
                "type": "color",
                "width_px": 512,
                "height_px": 512,
                "frame_rate": 30,
                "interleaved": "yes",
            },
        ],
    }, "viam:luxonis:oak-ffc-3p"),
   'attribute must be a bool_value'
)

incorrect_configs_and_errs = [
    empty_sensors_config,
    over_three_sensors_config,
    camera_sensors_wrong_type_config,
    two_cams_one_socket_config,
    one_mono_config,
    three_mono_config,
    invalid_sensor_type_config,
    invalid_socket_config,
    conflicting_sockets_config,
    missing_width_config,
    missing_height_config,
    dimension_not_num_type_config,
    dimension_not_whole_num_config,
    dimension_is_zero_config,
    dimension_is_neg_config,
    frame_rate_not_num_type,
    frame_rate_is_zero,
    frame_rate_is_neg,
    non_string_color_order_config,
    wrong_string_color_order_config,
    non_bool_interleaved_config
]

@pytest.mark.parametrize("invalid_config,msg", incorrect_configs_and_errs)
def test_invalid_configs(invalid_config, msg):
    with pytest.raises(ValidationError) as exc_info:
        Oak.validate(invalid_config)
    assert exc_info.type == ValidationError
    assert msg in str(exc_info.value)

### TEST VALID CONFIGS

one_color_config = make_component_config({
    "camera_sensors": [
        {
            "socket": "cam_a",
            "type": "color",
            "width_px": 512,
            "height_px": 512,
            "interleaved": False,
        }
    ],
}, "viam:luxonis:oak-ffc-3p")

two_color_config = make_component_config({
    "camera_sensors": [
        {
            "socket": "cam_b",
            "type": "color",
            "width_px": 512,
            "height_px": 512,
            "frame_rate": 30,
        },
        {
            "socket": "cam_c",
            "type": "color",
            "width_px": 512,
            "height_px": 512,
            "frame_rate": 30,
            "interleaved": False,
        },
    ],
}, "viam:luxonis:oak-ffc-3p")

three_color_config = make_component_config({
    "camera_sensors": [
        {
            "socket": "cam_a",
            "type": "color",
            "width_px": 512,
            "height_px": 512,
        },
        {
            "socket": "cam_b",
            "type": "color",
            "width_px": 512,
            "height_px": 512,
        },
        {
            "socket": "cam_c",
            "type": "color",
            "width_px": 512,
            "height_px": 512,
        },
    ],
}, "viam:luxonis:oak-ffc-3p")

# primary sensor is color
one_color_two_mono_config_1 = make_component_config({
    "camera_sensors": [
        {
            "socket": "cam_a",
            "type": "color",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "color_order": "bgr",
            "interleaved": False,
        },
        {
            "socket": "cam_b",
            "type": "depth",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "interleaved": False,
        },
        {
            "socket": "cam_c",
            "type": "depth",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "interleaved": False,
        },
    ],
}, "viam:luxonis:oak-ffc-3p")

# primary sensor is depth
one_color_two_mono_config_2 = make_component_config({
    "camera_sensors": [
        {
            "socket": "cam_b",
            "type": "depth",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "interleaved": False,
        },
        {
            "socket": "cam_c",
            "type": "depth",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "interleaved": False,
        },
        {
            "socket": "cam_a",
            "type": "color",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "color_order": "bgr",
            "interleaved": False,
        },
    ],
}, "viam:luxonis:oak-ffc-3p")

# primary sensor is depth
two_mono_config = make_component_config({
    "camera_sensors": [
        {
            "socket": "cam_b",
            "type": "depth",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "interleaved": True,
        },
        {
            "socket": "cam_c",
            "type": "depth",
            "width_px": 640,
            "height_px": 480,
            "frame_rate": 30,
            "interleaved": True,
        }
    ],
}, "viam:luxonis:oak-ffc-3p")

valid_configs = [
    one_color_config,
    two_color_config,
    three_color_config,
    one_color_two_mono_config_1,
    one_color_two_mono_config_2,
    two_mono_config,
]

@pytest.mark.parametrize("valid_config", valid_configs)
def test_valid_configs(valid_config):
    try:
        Oak.validate(valid_config)
    except Exception as e:
        s = (f"Expected a valid config to not raise {type(e)} during validation, yet it did: {e}")
        pytest.fail(reason=s)
