# Standard library
import logging
import struct
import time
from typing import ClassVar, Mapping, Any, Dict, Optional, Tuple, Literal, List, NamedTuple, Union
from typing_extensions import Self

# Third party
from PIL import Image

# Viam module
from viam.errors import ValidationError, ViamError
from viam.logging import getLogger
from viam.module.types import Reconfigurable, Stoppable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

# Viam camera
from viam.components.camera import Camera, DistortionParameters, IntrinsicParameters, RawImage
from viam.media.video import NamedImage, CameraMimeType

# OAK-D module
from src.worker import Worker

LOGGER = getLogger(__name__)

VALID_ATTRIBUTES = ['height_px', 'width_px', 'sensors', 'frame_rate', 'debug']

MAX_FPS = 60
MAX_WIDTH = 1280
MAX_HEIGHT = 720

DEFAULT_FRAME_RATE = 30
DEFAULT_WIDTH = MAX_WIDTH
DEFAULT_HEIGHT = MAX_HEIGHT
DEFAULT_IMAGE_MIMETYPE = CameraMimeType.JPEG
DEPTH_MIMETYPE = CameraMimeType.VIAM_RAW_DEPTH
DEFAULT_DEBUGGING = False

COLOR_SENSOR = 'color'
DEPTH_SENSOR = 'depth'

class OakDModel(Camera, Reconfigurable, Stoppable):
    '''
    OakDModel represents a physical OAK-D camera that can capture frames.
    '''
    class Properties(NamedTuple):
        intrinsic_parameters: IntrinsicParameters
        '''The properties of the camera'''
        distortion_parameters: DistortionParameters
        '''The distortion parameters of the camera'''
        supports_pcd: bool = True
        '''Whether the camera has a valid implementation of ``get_point_cloud``'''
    

    MODEL: ClassVar[Model] = Model(ModelFamily('viam', 'camera'), 'oak-d')
    worker: ClassVar[Worker]
    '''Singleton ``worker`` handles camera logic in a separate thread'''
    camera_properties: Camera.Properties

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        camera_cls = cls(config.name)
        camera_cls.validate(config)
        camera_cls.reconfigure(config, dependencies)
        return camera_cls

    @classmethod
    def validate(cls, config: ComponentConfig) -> None:
        attribute_map = config.attributes.fields

        # Helper that handles invalid attribute logic
        def handle_error(err_msg: str):
            LOGGER.error(f'Config attribute error: {err_msg}')
            try:
                cls.worker.stop()  # stop worker if active
            except AttributeError:
                pass
            raise ValidationError("Invalid config attribute.")
        
        # Helper that validates the attribute's value type
        def validate_type(
                attribute: str,
                expected_type: Literal['null_value', 'number_value', 'string_value', 'bool_value', 'struct_value', 'list_value'],
                ):
            value = attribute_map.get(key=attribute, default=None)
            if value.WhichOneof('kind') != expected_type:
                handle_error(f'the "{attribute}" attribute must be a {expected_type}, not {value}.')

        # Helper that validates height and width
        def validate_dimension(attribute: str, max_value: int):
            value = attribute_map.get(key=attribute, default=None)
            if value is None:
                return False  # user did not supply given dimension

            validate_type(attribute, 'number_value')
            number_value = value.number_value
            int_value = int(number_value)
            if int_value != number_value:
                handle_error(f'"{attribute}" must be a whole number.')
            if int_value > max_value:
                handle_error(f'inputted "{attribute}" of {int_value} exceeds max "{attribute}" of {max_value}.')
            if int_value <= 0:
                handle_error(f'inputted "{attribute}" cannot be less than or equal to 0.')
            return True  # user supplied a valid dimension
        
        # Validate config keys
        for attribute in attribute_map.keys():
            if attribute not in VALID_ATTRIBUTES:
                handle_error(f'"{attribute}" is not a valid attribute i.e. {VALID_ATTRIBUTES}')

        # Validate debug
        validate_type('debug', 'bool_value')

        # Validate sensors list
        sensors_value = attribute_map.get(key='sensors', default=None)
        if sensors_value is None:
            handle_error('''
                        a "sensors" attribute of a list of sensor(s) to read data from is required
                        e.g. ["depth", "color"], with the first sensor in the list being the main sensor
                        that get_image uses.
                        ''')
        validate_type('sensors', 'list_value')
        sensor_list = list(sensors_value.list_value)
        if len(sensor_list) == 0:
            handle_error('"sensors" attribute list cannot be empty.')
        if len(sensor_list) > 2:
            handle_error('"sensors" attribute list exceeds max length of two.')
        for sensor in sensor_list:
            if sensor != COLOR_SENSOR and sensor != DEPTH_SENSOR:
                handle_error(f'''
                            unknown sensor type "{sensor}" found in "sensors" attribute list.
                            Valid sensors include: "{COLOR_SENSOR}" and "{DEPTH_SENSOR}"
                            ''')
                
        # Validate frame rate
        frame_rate = attribute_map.get(key='frame_rate', default=None)
        if frame_rate:
            if frame_rate.WhichOneof('kind') != 'number_value':
                handle_error(f'the "frame_rate" attribute must be a number value, not {frame_rate}.')
            if frame_rate.number_value > 60 or frame_rate.number_value <= 0:
                handle_error(f'"frame_rate" must be a number > 0 and <= 60.')

        # Validate height
        contains_height = validate_dimension('height_px', MAX_HEIGHT)
        
        # Validate width
        contains_width = validate_dimension('width_px', MAX_WIDTH)

        # Validate dimensions together
        if (contains_height and not contains_width) or (contains_width and not contains_height):
            handle_error('received only one dimension attribute. Please supply both "height_px" and "width_px", or neither.')

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        cls = type(self)
        try:
            LOGGER.debug('Trying to stop worker.')
            cls.worker.stop()
            LOGGER.info('Stopping active worker and reconfiguring...')
        except AttributeError:
            LOGGER.debug('No active worker.')

        self.camera_properties = Camera.Properties(
                supports_pcd=False,
                distortion_parameters=None,
                intrinsic_parameters=None
            )
        attribute_map = config.attributes.fields
        self.debugging = attribute_map['debug'].bool_value or DEFAULT_DEBUGGING
        if self.debugging:
            LOGGER.setLevel(logging.DEBUG)
            LOGGER.debug(f'Running module in debugging mode.')
        self.sensors = list(attribute_map['sensors'].list_value)
        LOGGER.debug(f'Set sensors attr to {self.sensors}')
        self.height = int(attribute_map['height_px'].number_value) or DEFAULT_HEIGHT
        LOGGER.debug(f'Set height attr to {self.height}')
        self.width = int(attribute_map['width_px'].number_value) or DEFAULT_WIDTH
        LOGGER.debug(f'Set width attr to {self.width}')
        self.frame_rate = attribute_map['frame_rate'].number_value or DEFAULT_FRAME_RATE
        LOGGER.debug(f'Set frame_rate attr to {self.frame_rate}')

        should_get_color, should_get_depth = COLOR_SENSOR in self.sensors, DEPTH_SENSOR in self.sensors
        callback = lambda: self.reconfigure(config, dependencies)
        cls.worker = Worker(height=self.height,
                            width=self.width,
                            frame_rate=self.frame_rate,
                            should_get_color=should_get_color,
                            should_get_depth=should_get_depth,
                            reconfigure=callback,
                            logger=LOGGER)
        cls.worker.start()

    # Implements ``stop`` under the Stoppable protocol to free resources
    def stop(self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs):
        cls = type(self)
        cls.worker.stop()

    async def get_image(
        self, mime_type: str = '', *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Union[Image.Image, RawImage]:
        '''Get the next image from the camera as an Image or RawImage.
        Be sure to close the image when finished.

        NOTE: If the mime type is ``image/vnd.viam.dep`` you can use :func:`viam.media.video.RawImage.bytes_to_depth_array`
        to convert the data to a standard representation.

        Args:
            mime_type (str): The desired mime type of the image. This does not guarantee output type

        Returns:
            Image | RawImage: The frame
        '''
        # LOGGER.debug('Handling get_image request.')
        cls = type(self)
        main_sensor = self.sensors[0]
        if main_sensor == COLOR_SENSOR:
            return Image.fromarray(cls.worker.get_color_image(), 'RGB')
        if main_sensor == DEPTH_SENSOR:
            return Image.fromarray(cls.worker.get_depth_map(), 'I;16').convert('RGB')
        LOGGER.error('get_image failed due to misconfigured `sensors` attribute.')

    
    async def get_images(self, *, timeout: Optional[float] = None, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        '''Get simultaneous images from different imagers, along with associated metadata.
        This should not be used for getting a time series of images from the same imager.

        Returns:
            Tuple[List[NamedImage], ResponseMetadata]:
                - List[NamedImage]:
                  The list of images returned from the camera system.

                - ResponseMetadata:
                  The metadata associated with this response
        '''
        raise Exception('under construction...')
    
    async def get_point_cloud(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[bytes, str]:
        '''
        Get the next point cloud from the camera. This will be
        returned as bytes with a mimetype describing
        the structure of the data. The consumer of this call
        should encode the bytes into the formatted suggested
        by the mimetype.

        To deserialize the returned information into a numpy array, use the Open3D library.
        ::

            import numpy as np
            import open3d as o3d

            data, _ = await camera.get_point_cloud()

            # write the point cloud into a temporary file
            with open('/tmp/pointcloud_data.pcd', 'wb') as f:
                f.write(data)
            pcd = o3d.io.read_point_cloud('/tmp/pointcloud_data.pcd')
            points = np.asarray(pcd.points)

        Returns:
            bytes: The pointcloud data.
            str: The mimetype of the pointcloud (e.g. PCD).
        '''
        if DEPTH_SENSOR not in self.sensors:
            details = 'Please include "depth" in the "sensors" attribute list.'
            raise MethodNotAllowed('get_point_cloud', details)
        cls = type(self)
        return (cls.worker.get_pcd().tobytes(), CameraMimeType.PCD)

    
    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Properties:
        '''
        Get the camera intrinsic parameters and camera distortion parameters

        Returns:
            Properties: The properties of the camera
        '''
        return self.camera_properties

    # possibly use later to encode bytes to raw for depth data
    def _encode_depth_raw(self, data: bytes, little_endian):
        DEPTH_MAGIC_NUMBER = struct.pack('>Q', 4919426490892632400)  # UTF-8 binary encoding for 'DEPTHMAP', big-endian
        DEPTH_MAGIC_BYTE_COUNT = struct.calcsize('Q')  # Number of bytes used to represent the depth magic number
        DEPTH_WIDTH_BYTE_COUNT = struct.calcsize('Q')  # Number of bytes used to represent depth image width
        DEPTH_HEIGHT_BYTE_COUNT = struct.calcsize('Q')  # Number of bytes used to represent depth image height
        if self.debugging:
            start = time.time()

        # Depth header contains 8 bytes for the magic number, followed by 8 bytes for width and 8 bytes for height. Each pixel has 2 bytes.
        pixel_byte_count = 2 * self.width * self.height
        width_to_encode = struct.pack('>Q', self.width)  # Convert width to big-endian
        height_to_encode = struct.pack('>Q', self.height)  # Convert height to big-endian
        total_byte_count = DEPTH_MAGIC_BYTE_COUNT + DEPTH_WIDTH_BYTE_COUNT + DEPTH_HEIGHT_BYTE_COUNT + pixel_byte_count

        # Create a bytearray to store the encoded data
        raw_buf = bytearray(total_byte_count)

        offset = 0

        # Copy the depth magic number into the buffer
        raw_buf[offset:offset + DEPTH_MAGIC_BYTE_COUNT] = DEPTH_MAGIC_NUMBER
        offset += DEPTH_MAGIC_BYTE_COUNT

        # Copy the encoded width and height into the buffer
        raw_buf[offset:offset + DEPTH_WIDTH_BYTE_COUNT] = width_to_encode
        offset += DEPTH_WIDTH_BYTE_COUNT
        raw_buf[offset:offset + DEPTH_HEIGHT_BYTE_COUNT] = height_to_encode
        offset += DEPTH_HEIGHT_BYTE_COUNT

        if little_endian:
            # Copy the data as is
            raw_buf[offset:offset + pixel_byte_count] = data
        else:
            pixel_offset = 0
            for _ in range(self.width * self.height):
                pix = struct.unpack_from('<H', data, pixel_offset)[0]
                pix_encode = struct.pack('>H', pix)  # Convert pixel value to big-endian
                raw_buf[offset:offset + 2] = pix_encode
                pixel_offset += 2
                offset += 2

        if self.debugging:
            stop = time.time()
            duration = int((stop - start) * 1000)
            LOGGER.debug(f'[GetImage] RAW depth encode: {duration}ms')

        return bytes(raw_buf)

class MethodNotAllowed(ViamError):
    """
    Exception raised when attempting to call a method
    with a configuration that does not support said method.
    """
    def __init__(self, name: str, details: str) -> None:
        self.name = name
        self.message = f'Cannot invoke method "{name}" with current config. {details}'
        super().__init__(self.message)
