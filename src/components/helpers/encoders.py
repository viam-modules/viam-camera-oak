import io
import struct
from typing import List, Tuple

from google.protobuf.timestamp_pb2 import Timestamp
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from viam.logging import getLogger
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.common import ResponseMetadata

from src.config import OakDConfig
from src.components.helpers.shared import CapturedData

LOGGER = getLogger("viam-oak-encoders-logger")


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
    If color information is available (6 columns: x, y, z, r, g, b), it packs the color into a single float field.

    Args:
        arr (NDArray): points data (shape Nx3 for uncolored or Nx6 for colored)
    Returns:
        Tuple[bytes, CameraMimeType]: PCD bytes and associated mime type
    """
    # Convert to meters and to float32
    # DepthAI examples indicate that we need such a normalization for the points data
    # https://github.com/luxonis/depthai/blob/f4a0d3d4364565faacf3ce9f131a42b2b951ec1b/depthai_sdk/src/depthai_sdk/visualize/visualizers/viewer_visualizer.py#L72
    points_xyz = (arr[:, :3] / 1000.0).astype(np.float32)
    if arr.shape[1] == 3:
        version = "VERSION .7\n"
        fields = "FIELDS x y z\n"
        size = "SIZE 4 4 4\n"
        type_of = "TYPE F F F\n"
        count = "COUNT 1 1 1\n"
        height = "HEIGHT 1\n"
        viewpoint = "VIEWPOINT 0 0 0 1 0 0 0\n"
        width = f"WIDTH {points_xyz.shape[0]}\n"
        points_count = f"POINTS {points_xyz.shape[0]}\n"
        data = "DATA binary\n"
        header = f"{version}{fields}{size}{type_of}{count}{width}{height}{viewpoint}{points_count}{data}"
        header_bytes = bytes(header, "UTF-8")
        float_array = np.array(points_xyz, dtype="f")
        return (header_bytes + float_array.tobytes(), CameraMimeType.PCD)
    elif arr.shape[1] == 6:
        colors = arr[:, 3:6].astype(np.uint8).astype(np.uint32)
        rgb_int = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        rgb_float = rgb_int.view(np.float32)

        # Concatenate the xyz coordinates with the packed rgb value
        colored_points = np.column_stack((points_xyz, rgb_float))

        version = "VERSION .7\n"
        fields = "FIELDS x y z rgb\n"
        size = "SIZE 4 4 4 4\n"
        type_of = "TYPE F F F F\n"
        count = "COUNT 1 1 1 1\n"
        height = "HEIGHT 1\n"
        viewpoint = "VIEWPOINT 0 0 0 1 0 0 0\n"
        width = f"WIDTH {colored_points.shape[0]}\n"
        points_count = f"POINTS {colored_points.shape[0]}\n"
        data = "DATA binary\n"
        header = f"{version}{fields}{size}{type_of}{count}{width}{height}{viewpoint}{points_count}{data}"
        header_bytes = bytes(header, "UTF-8")

        return (header_bytes + colored_points.tobytes(), CameraMimeType.PCD)
    else:
        raise ValueError(
            f"Unexpected point cloud array shape with {arr.shape[1]} columns"
        )


def convert_seconds_float_to_metadata(seconds_float: float) -> ResponseMetadata:
    """
    Converts a float representing seconds since epoch to a Viam ResponseMetadata
    proto msg object.

    Args:
        seconds_float (float): seconds with decimals

    Returns:
        ResponseMetadata: ResponseMetadata obj
    """
    seconds_int = int(seconds_float)
    nanoseconds_int = int((seconds_float - seconds_int) * 1e9)
    metadata = ResponseMetadata(
        captured_at=Timestamp(seconds=seconds_int, nanos=nanoseconds_int)
    )
    return metadata
