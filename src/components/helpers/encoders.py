import io
import struct
from typing import Tuple

from google.protobuf.timestamp_pb2 import Timestamp
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.common import ResponseMetadata

from src.components.helpers.shared import CapturedData

LOGGER = getLogger("oak-encoders-logger")


def encode_depth_raw(data: bytes, shape: Tuple[int, int]) -> bytes:
    """
    Encodes raw data into a bytes payload deserializable by the Viam SDK (camera mime type depth)

    Args:
        data (bytes): raw bytes
        shape (Tuple[int, int]): output dimensions of depth map (height x width)

    Returns:
        bytes: encoded bytes
    """
    height, width = shape  # using np shape for actual output's height/width
    MAGIC_NUMBER = struct.pack(
        ">Q", 4919426490892632400
    )  # UTF-8 encoding for 'DEPTHMAP'
    MAGIC_BYTE_COUNT = struct.calcsize("Q")
    WIDTH_BYTE_COUNT = struct.calcsize("Q")
    HEIGHT_BYTE_COUNT = struct.calcsize("Q")

    # Depth header contains 8 bytes for the magic number, followed by 8 bytes for width and 8 bytes for height. Each pixel has 2 bytes.
    pixel_byte_count = np.dtype(np.uint16).itemsize * width * height
    width_to_encode = struct.pack(">Q", width)  # Q signifies big-endian
    height_to_encode = struct.pack(">Q", height)
    total_byte_count = (
        MAGIC_BYTE_COUNT + WIDTH_BYTE_COUNT + HEIGHT_BYTE_COUNT + pixel_byte_count
    )
    LOGGER.debug(
        f"Calculated size:  {MAGIC_BYTE_COUNT} + {WIDTH_BYTE_COUNT} + {HEIGHT_BYTE_COUNT} + {pixel_byte_count} = {total_byte_count}"
    )
    LOGGER.debug(f"Actual data size: {len(data)}")

    # Create a bytearray to store the encoded data
    raw_buf = bytearray(total_byte_count)
    offset = 0

    # Copy the depth magic number into the buffer
    raw_buf[offset : offset + MAGIC_BYTE_COUNT] = MAGIC_NUMBER
    offset += MAGIC_BYTE_COUNT

    # Copy the encoded width and height into the buffer
    raw_buf[offset : offset + WIDTH_BYTE_COUNT] = width_to_encode
    offset += WIDTH_BYTE_COUNT
    raw_buf[offset : offset + HEIGHT_BYTE_COUNT] = height_to_encode
    offset += HEIGHT_BYTE_COUNT

    # Copy data into rest of the buffer
    raw_buf[offset : offset + pixel_byte_count] = data
    return bytes(raw_buf)


def encode_jpeg_bytes(arr: NDArray, is_depth: bool = False) -> bytes:
    """
    Encodes numpy color image data into bytes decodable in the JPEG image format.

    Args:
        arr (NDArray): frame
        is_depth (bool): whether arr is depth data or RGB data

    Returns:
        bytes: JPEG bytes
    """
    if is_depth:
        if arr.dtype != np.uint16:
            raise ValueError("Depth data should be of type uint16")
        pil_image = Image.fromarray(arr, mode="I;16").convert("RGB")
    else:
        if arr.dtype != np.uint8:
            raise ValueError("RGB data should be of type uint8")
        pil_image = Image.fromarray(arr)

    output_buffer = io.BytesIO()
    pil_image.save(output_buffer, format="JPEG")
    raw_bytes = output_buffer.getvalue()
    output_buffer.close()
    return raw_bytes


def encode_pcd(arr: NDArray):
    """
    Encodes numpy points data into bytes decodable in the PCD format.

    Args:
        arr (NDArray): points data

    Returns:
        bytes: PCD bytes
    """
    # DepthAI examples indicate that we need to normalize data by / 1000
    # https://github.com/luxonis/depthai/blob/f4a0d3d4364565faacf3ce9f131a42b2b951ec1b/depthai_sdk/src/depthai_sdk/visualize/visualizers/viewer_visualizer.py#L72
    flat_array = arr.reshape(-1, arr.shape[-1]) / 1000.0
    version = "VERSION .7\n"
    fields = "FIELDS x y z\n"
    size = "SIZE 4 4 4\n"
    type_of = "TYPE F F F\n"
    count = "COUNT 1 1 1\n"
    height = "HEIGHT 1\n"
    viewpoint = "VIEWPOINT 0 0 0 1 0 0 0\n"
    data = "DATA binary\n"
    width = f"WIDTH {len(flat_array)}\n"
    points = f"POINTS {len(flat_array)}\n"
    header = f"{version}{fields}{size}{type_of}{count}{width}{height}{viewpoint}{points}{data}"
    header_bytes = bytes(header, "UTF-8")
    float_array = np.array(flat_array, dtype="f")
    return (header_bytes + float_array.tobytes(), CameraMimeType.PCD)


def make_metadata_from_seconds_float(seconds_float: float) -> ResponseMetadata:
    seconds_int = int(seconds_float)
    nanoseconds_int = int((seconds_float - seconds_int) * 1e9)
    metadata = ResponseMetadata(
        captured_at=Timestamp(seconds=seconds_int, nanos=nanoseconds_int)
    )
    return metadata


def handle_synced_color_and_depth(color_data: CapturedData, depth_data: CapturedData):
    images = []
    arr, captured_at = color_data.np_array, color_data.captured_at
    jpeg_encoded_bytes = encode_jpeg_bytes(arr)
    img = NamedImage("color", jpeg_encoded_bytes, CameraMimeType.JPEG)
    images.append(img)

    arr, captured_at = depth_data.np_array, depth_data.captured_at
    depth_encoded_bytes = encode_depth_raw(arr.tobytes(), arr.shape)
    img = NamedImage("depth", depth_encoded_bytes, CameraMimeType.VIAM_RAW_DEPTH)
    images.append(img)

    metadata = make_metadata_from_seconds_float(seconds_float=captured_at)
    return images, metadata
