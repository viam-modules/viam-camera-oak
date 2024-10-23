from numpy.typing import NDArray

from depthai import CameraBoardSocket


def get_socket_from_str(s: str) -> CameraBoardSocket:
    """
    Socket str -> CameraBoardSocket object

    Args:
        s (str): socket str e.g. "cam_a", "cam_b" etc.

    Raises:
        Exception: if socket str is unrecognized or unsupported

    Returns:
        CameraBoardSocket: CameraBoardSocket DepthAI object
    """
    if s == "cam_a":
        return CameraBoardSocket.CAM_A
    elif s == "cam_b":
        return CameraBoardSocket.CAM_B
    elif s == "cam_c":
        return CameraBoardSocket.CAM_C
    else:
        raise Exception(f"Camera socket '{s}' is not recognized or supported.")


class CapturedData:
    """
    CapturedData is image data with the data as an np array,
    plus the timestamp it was captured at.

    This class uses __slots__ to:
        - Optimize memory usage by pre-defining attributes.
        - Speed up attribute access slightly.
        - Prevent unidiomatic additions of new attributes at runtime.
    """

    __slots__ = ("np_array", "captured_at")

    def __init__(self, np_array: NDArray, captured_at: float) -> None:
        self.np_array = np_array
        self.captured_at = captured_at
