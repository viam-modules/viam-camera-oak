from typing import Sequence
import pytest

from viam.errors import ValidationError

from src.services.yolo_detection_network import YoloDetectionNetwork
from tests.helpers import make_service_config

MODEL_STR = "viam:luxonis:yolo-detection-network"

missing_input_source = (
    make_service_config({
    }, MODEL_STR),
    '"input_source" is a required field, but was not detected'
)

invalid_input_source_type = (
    make_service_config({
        "input_source": 123,
    }, MODEL_STR),
    '"input_source" attribute must be a string_value'
)

invalid_input_source_str = (
    make_service_config({
        "input_source": "foo",
    }, MODEL_STR),
    '"input_source" attribute must be either'
)

missing_required_dimensions = (
    make_service_config({
        "input_source": "cam_a",
    }, MODEL_STR),
    '"width_px" is a required field, but was not detected'
)

dimension_not_number_value = (
    make_service_config({
        "input_source": "cam_b",
        "width_px": "1280px",
        "height_px": 720
    }, MODEL_STR),
    '"width_px" attribute must be a number_value'
)

dimension_not_whole_number = (
    make_service_config({
        "input_source": "cam_c",
        "width_px": 1280.5,
        "height_px": 720
    }, MODEL_STR),
    '"width_px" must be a whole number'
)

invalid_num_threads_value = (
    make_service_config({
        "input_source": "color",
        "width_px": 1280,
        "height_px": 720,
        "num_threads": 3
    }, MODEL_STR),
    '"num_threads" must be 0, 1, or 2. 0 means AUTO. You set 3'
)

invalid_num_nce_value = (
    make_service_config({
        "input_source": "depth",
        "width_px": 1280,
        "height_px": 720,
        "num_nce_per_thread": 3
    }, MODEL_STR),
    '"num_nce_per_thread" must be 1 or 2. You set 3'
)

invalid_is_object_tracker_value = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "is_object_tracker": "yes"
    }, MODEL_STR),
    '"is_object_tracker" attribute must be a bool_value'
)

missing_yolo_config = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720
    }, MODEL_STR),
    '"yolo_config" is a required field, but was not detected'
)

missing_blob_path = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {}
    }, MODEL_STR),
    '"blob_path" is a required field, but was not detected'
)

empty_blob_path = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": ""}
    }, MODEL_STR),
    '"blob_path" cannot be empty string.'
)

labels_not_list = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": "label1"}
    }, MODEL_STR),
    '"labels" attribute must be a list_value'
)

empty_labels_list = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": []}
    }, MODEL_STR),
    '"labels" attribute list cannot be empty.'
)

non_string_labels = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": [123]}
    }, MODEL_STR),
    'List elements of "labels" must be strings.'
)

empty_str_labels = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": [""]}
    }, MODEL_STR),
    'List elements of "labels" cannot be empty strings.'
)

invalid_confidence_threshold = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "confidence_threshold": 1.5}
    }, MODEL_STR),
    '"confidence_threshold" must be between 0 and 1. You set 1.5'
)

invalid_iou_threshold = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "iou_threshold": 1.5}
    }, MODEL_STR),
    '"iou_threshold" must be between 0 and 1. You set 1.5'
)

anchors_not_list = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": "123"}
    }, MODEL_STR),
    '"anchors" attribute must be a list_value'
)

empty_anchors_list = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": []}
    }, MODEL_STR),
    '"labels" attribute list cannot be empty.'
)

non_number_anchors = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": ["string"]}
    }, MODEL_STR),
    'List elements of "anchors" must be integers.'
)

negative_anchor_value = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [-1]}
    }, MODEL_STR),
    'List elements of "anchors" cannot be negative.'
)

invalid_anchor_masks = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "anchor_masks": "mask"}
    }, MODEL_STR),
    '"anchor_masks" attribute must be a struct_value'
)

empty_anchor_masks = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "anchor_masks": {}}
    }, MODEL_STR),
    'No anchor masks found in "anchor_masks". Please either supply them or remove the attribute entirely.'
)

anchor_mask_not_list = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "anchor_masks": {"mask1": "not_list"}}
    }, MODEL_STR),
    'Anchor mask subfield "mask1" must be a list value.'
)

empty_anchor_mask_list = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "anchor_masks": {"mask1": []}}
    }, MODEL_STR),
    'Anchor mask list subfield "mask1" cannot be empty.'
)

non_number_anchor_mask_elements = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "anchor_masks": {"mask1": ["string"]}}
    }, MODEL_STR),
    'List elements of "anchors" must be integers.'
)

negative_anchor_mask_value = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "anchor_masks": {"mask1": [-1]}}
    }, MODEL_STR),
    'List elements of "anchors" cannot be negative.'
)

invalid_coordinate_size = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "coordinate_size": -1}
    }, MODEL_STR),
    '"coordinate_size" attribute must be positive, not -1'
)

missing_cam_name = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "coordinate_size": 4}
    }, MODEL_STR),
    '"cam_name" is a required field, but was not detected'
)

empty_str_cam_name = (
    make_service_config({
        "input_source": "cam_a",
        "width_px": 1280,
        "height_px": 720,
        "yolo_config": {"blob_path": "/path/to/blob", "labels": ["label1"], "anchors": [1, 2, 3], "coordinate_size": 4},
        "cam_name": ""
    }, MODEL_STR),
    '"cam_name" attribute cannot be empty string.'
)

configs_and_msgs = [
    missing_input_source,
    invalid_input_source_type,
    invalid_input_source_str,
    missing_required_dimensions,
    dimension_not_number_value,
    dimension_not_whole_number,
    invalid_num_threads_value,
    invalid_num_nce_value,
    invalid_is_object_tracker_value,
    missing_yolo_config,
    missing_blob_path,
    empty_blob_path,
    labels_not_list,
    empty_labels_list,
    non_string_labels,
    empty_str_labels,
    invalid_confidence_threshold,
    invalid_iou_threshold,
    anchors_not_list,
    empty_anchors_list,
    non_number_anchors,
    negative_anchor_value,
    invalid_anchor_masks,
    empty_anchor_masks,
    anchor_mask_not_list,
    empty_anchor_mask_list,
    non_number_anchor_mask_elements,
    negative_anchor_mask_value,
    invalid_coordinate_size,
    missing_cam_name,
    empty_str_cam_name
]

full_correct_config = make_service_config({
    "input_source": "cam_a",
    "width_px": 1280,
    "height_px": 720,
    "num_threads": 1,
    "num_nce_per_thread": 2,
    "is_object_tracker": True,
    "yolo_config": {
        "blob_path": "/path/to/blob",
        "labels": ["label1, label2"],
        "confidence_threshold": 0.5,
        "iou_threshold": 0.5,
        "anchors": [1, 2, 3],
        "anchor_masks": {
            "mask1": [1, 2, 3]
        },
        "coordinate_size": 4
    },
    "cam_name": "oak-cam"
}, MODEL_STR)

minimal_correct_config = make_service_config({
    "input_source": "cam_a",
    "width_px": 1280,
    "height_px": 720,
    "yolo_config": {
        "blob_path": "/path/to/blob",
        "labels": ["label1", "label2"]
    },
    "cam_name": "oak-cam"
}, MODEL_STR)

@pytest.mark.parametrize("config,msg", configs_and_msgs)
def test_validate_errors_parameterized(config, msg):
    with pytest.raises(ValidationError) as exc_info:
        YoloDetectionNetwork.validate(config)
        assert exc_info.type == ValidationError
    assert msg in str(exc_info.value)

def test_validate_no_errors():
    def check_deps(deps: Sequence[str]):
        assert len(deps) == 1
        cam_name = deps[0]
        assert isinstance(cam_name, str)
        assert cam_name == "oak-cam"

    try:
        ydn = YoloDetectionNetwork
        deps1 = YoloDetectionNetwork.validate(full_correct_config)
        check_deps(deps1)
        ydn.reconfigure(ydn, full_correct_config, deps1)

        deps2 = YoloDetectionNetwork.validate(minimal_correct_config)
        check_deps(deps2)
        ydn.reconfigure(ydn, minimal_correct_config, deps2)
    except Exception as e:
        s = (f"Expected a correct config to not raise {type(e)} during validation/reconfiguration, yet it did :,)")
        pytest.fail(reason=s)
