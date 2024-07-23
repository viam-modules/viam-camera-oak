from collections import OrderedDict
import io
from logging import Logger
from threading import Lock
import struct
import time
from typing import Dict, Literal, Optional, OrderedDict, Tuple
from depthai_sdk.classes.packets import BasePacket
from depthai_sdk.classes.packet_handlers import QueuePacketHandler
from numpy.typing import NDArray
import numpy as np
from PIL import Image


class MessageSynchronizer:
    """
    MessageSynchronizer manages synchronization of frame messages for color and depth data from OakCamera packet queues,
    maintaining an ordered dictionary of messages keyed chronologically by sequence number.
    """

    MAX_MSGS_SIZE = 50
    msgs: OrderedDict[int, Dict[str, BasePacket]]
    write_lock: Lock

    def __init__(self):
        # msgs maps frame sequence number to a dictionary that maps frame_type (i.e. "color" or "depth") to a data packet
        self.msgs = OrderedDict()
        self.write_lock = Lock()

    def add_msg(
        self, msg: BasePacket, frame_type: Literal["color", "depth"], seq: int
    ) -> None:
        with self.write_lock:
            # Update recency if previously already stored in dict
            if seq in self.msgs:
                self.msgs.move_to_end(seq)

            self.msgs.setdefault(seq, {})[frame_type] = msg
            self._cleanup_msgs()

    def get_synced_msgs(self) -> Optional[Dict[str, BasePacket]]:
        for sync_msgs in self.msgs.values():
            if len(sync_msgs) == 2:  # has both color and depth
                return sync_msgs
        return None

    def _add_msgs_from_queue(
        self, frame_type: Literal["color", "depth"], queue_handler: QueuePacketHandler
    ) -> None:
        queue_obj = queue_handler.get_queue()
        with queue_obj.mutex:
            q_snapshot = list(queue_obj.queue)

        for msg in q_snapshot:
            self.add_msg(msg, frame_type, msg.get_sequence_num())

    def get_most_recent_msg(
        self, q_handler: QueuePacketHandler, frame_type: Literal["color", "depth"]
    ) -> Optional[BasePacket]:
        self._add_msgs_from_queue(frame_type, q_handler)
        while len(self.msgs) < 1:
            self._add_msgs_from_queue(frame_type, q_handler)
            time.sleep(0.1)
        # Traverse in reverse to get the most recent
        for msg_dict in reversed(self.msgs.values()):
            if frame_type in msg_dict:
                return msg_dict[frame_type]
        raise Exception(f"No message of type '{frame_type}' in frame queue.")

    def _cleanup_msgs(self):
        while len(self.msgs) > self.MAX_MSGS_SIZE:
            self.msgs.popitem(last=False)  # remove oldest item


class CapturedData:
    """
    CapturedData is image data with the data as an np array,
    plus the timestamp it was captured at.
    """

    def __init__(self, np_array: NDArray, captured_at: float) -> None:
        self.np_array = np_array
        self.captured_at = captured_at


def encode_depth_raw(data: bytes, shape: Tuple[int, int], logger: Logger) -> bytes:
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
    logger.debug(
        f"Calculated size:  {MAGIC_BYTE_COUNT} + {WIDTH_BYTE_COUNT} + {HEIGHT_BYTE_COUNT} + {pixel_byte_count} = {total_byte_count}"
    )
    logger.debug(f"Actual data size: {len(data)}")

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

    with io.BytesIO() as output_buffer:
        pil_image.save(output_buffer, format="JPEG")
        raw_bytes = output_buffer.getvalue()
    return raw_bytes
