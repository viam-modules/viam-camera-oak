import numpy as np
from typing import Dict, List, Mapping, Tuple
from uuid import UUID
import base64

import depthai as dai

from viam.media.video import CameraMimeType, ViamImage
from viam.proto.service.vision import Detection
from viam.utils import ValueTypes
from viam.services.vision import CaptureAllResult

from src.config import YDNConfig, Sensor

YDN_CONFIGURE = "ydn_configure"
YDN_DECONFIGURE = "ydn_deconfigure"
YDN_DETECTIONS = "ydn_detections"
YDN_CAPTURE_ALL = "ydn_capture_all"


def encode_ydn_configure_command(
    cfg: YDNConfig, service_name: str, service_id: UUID
) -> Dict[str, ValueTypes]:
    """Encodes the YDNConfig object into a format suitable for the `do_command` call."""

    command_dict = {
        "input_source": cfg.input_source,
        "width": cfg.width,
        "height": cfg.height,
        "num_threads": cfg.num_threads,
        "num_nce_per_thread": cfg.num_nce_per_thread,
        "is_object_tracker": cfg.is_object_tracker,
        "yolo_config": {
            "blob_path": cfg.blob_path,
            "labels": cfg.labels,
            "confidence_threshold": cfg.confidence_threshold,
            "iou_threshold": cfg.iou_threshold,
            "anchors": cfg.anchors,
            "anchor_masks": cfg.anchor_masks,
            "coordinate_size": cfg.coordinate_size,
        },
        "cmd": YDN_CONFIGURE,
        "sender_name": service_name,
        "sender_id": service_id.hex,
    }

    return command_dict


def decode_ydn_configure_command(command_dict: Mapping[str, ValueTypes]) -> YDNConfig:
    """Decodes a command dictionary into a YDNConfig object."""

    yolo_config_dict = command_dict["yolo_config"]
    cfg = YDNConfig.from_kwargs(
        input_source=command_dict["input_source"],
        width=int(command_dict["width"]),
        height=int(command_dict["height"]),
        num_threads=int(command_dict["num_threads"]),
        num_nce_per_thread=int(command_dict["num_nce_per_thread"]),
        is_object_tracker=command_dict["is_object_tracker"],
        blob_path=yolo_config_dict["blob_path"],
        labels=yolo_config_dict["labels"],
        confidence_threshold=yolo_config_dict["confidence_threshold"],
        iou_threshold=yolo_config_dict["iou_threshold"],
        anchors=yolo_config_dict["anchors"],
        anchor_masks={k: v for k, v in yolo_config_dict["anchor_masks"].items()},
        coordinate_size=int(yolo_config_dict["coordinate_size"]),
        service_name=command_dict["sender_name"],
        service_id=command_dict["sender_id"],
    )
    return cfg


def encode_ydn_deconfigure_command(
    service_id: UUID,
) -> Dict[str, ValueTypes]:
    command_dict = {
        "cmd": YDN_DECONFIGURE,
        "sender_id": service_id.hex,
    }
    return command_dict


def _normalize_bbox(
    width: int, height: int, xmin: float, ymin: float, xmax: float, ymax: float
) -> Tuple[int, int, int, int]:
    normVals = np.array([width, height, width, height])
    bbox = np.array([xmin, ymin, xmax, ymax])
    normalized_bbox = (np.clip(bbox, 0, 1) * normVals).astype(int)
    return tuple(normalized_bbox)


def encode_detections(
    dets: dai.ImgDetections, labels: List[str], sensor: Sensor
) -> List[Dict]:
    encoded_detections = []
    for det in dets.detections:
        xmin, ymin, xmax, ymax = _normalize_bbox(
            sensor.width, sensor.height, det.xmin, det.ymin, det.xmax, det.ymax
        )
        dictionary = {
            "x_min": xmin,
            "y_min": ymin,
            "x_max": xmax,
            "y_max": ymax,
            "confidence": det.confidence,
            "class_name": labels[det.label],
        }
        encoded_detections.append(dictionary)
    return encoded_detections


def decode_detections(det_list: List) -> List[Detection]:
    viam_dets: List[Detection] = []
    for det_dict in det_list:
        viam_det = Detection(
            x_min=int(det_dict["x_min"]),
            x_max=int(det_dict["x_max"]),
            y_min=int(det_dict["y_min"]),
            y_max=int(det_dict["y_max"]),
            confidence=det_dict["confidence"],
            class_name=det_dict["class_name"],
        )
        viam_dets.append(viam_det)
    return viam_dets


def encode_image_data(img_bytes: bytes) -> Dict:
    img_data_dict = {
        "bytes": base64.b64encode(img_bytes).decode("ascii"),
        "mime_type": CameraMimeType.JPEG.value,
    }
    return img_data_dict


def decode_image_data(img_dict: Dict) -> ViamImage:
    try:
        jpeg_bytes = base64.b64decode(img_dict["bytes"])
        mime_type_str = img_dict["mime_type"]
        mime_type = CameraMimeType.from_string(mime_type_str)
    except (KeyError, ValueError, base64.binascii.Error) as e:
        raise ValueError("Invalid image data dictionary") from e
    return ViamImage(jpeg_bytes, mime_type)


def convert_capture_all_dict_into_capture_all(d: Dict) -> CaptureAllResult:
    res = CaptureAllResult()
    if "image_data" in d:
        img_dict: Dict = d["image_data"]
        res.image = decode_image_data(img_dict)
    if "detections" in d:
        det_list: Dict = d["detections"]
        res.detections = decode_detections(det_list)
    return res
