# Standard library
import io
import logging
import struct
import threading
from typing import Any, Callable, ClassVar, Dict, Literal, List, Mapping, NamedTuple, Optional, Tuple, Union
from typing_extensions import Self

# Third party
from google.protobuf.timestamp_pb2 import Timestamp
import numpy as np
from PIL import Image

# Viam module
from viam.errors import NotSupportedError, ValidationError, ViamError
from viam.logging import getLogger, addHandlers
from viam.module.types import Reconfigurable, Stoppable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

# Viam camera
from viam.components.camera import Camera, DistortionParameters, IntrinsicParameters, RawImage
from viam.media.video import CameraMimeType, NamedImage

# OAK-D module
from src.worker import Worker


LOGGER = getLogger(__name__)

VALID_ATTRIBUTES = ['height_px', 'width_px', 'sensors', 'frame_rate', 'debug']

MAX_HEIGHT = 1080
MAX_WIDTH = 1920
MAX_FPS = 60

DEFAULT_FRAME_RATE = 30
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 400
DEFAULT_IMAGE_MIMETYPE = CameraMimeType.JPEG
DEFAULT_DEBUGGING = False

COLOR_SENSOR = 'color'
DEPTH_SENSOR = 'depth'

### TODO RSDK-5592: remove the below bandaid fix once https://github.com/luxonis/depthai/pull/1135 is in a new release
root_logger = logging.getLogger()

# Remove all handlers from the root logger
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Apply Viam's logging handlers
addHandlers(root_logger)
### TODO


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
        '''
        Used to register the module model class as a resource.

        Args:
            config (ComponentConfig)
            dependencies (Mapping[ResourceName, ResourceBase])

        Returns:
            Self: the OAK-D model class
        '''        
        camera_cls = cls(config.name)
        camera_cls.validate(config)
        camera_cls.reconfigure(config, dependencies)
        return camera_cls

    @classmethod
    def validate(cls, config: ComponentConfig) -> None:
        '''
        A procedure called in reconfigure to validate the robot config.

        Args:
            config (ComponentConfig)

        Raises:
            ValidationError: with a description of what is wrong about the config

        Returns:
            None
        '''        
        attribute_map = config.attributes.fields

        def handle_error(err_msg: str) -> None:
            '''
            handle_error is invoked when there is an error in validation.
            It logs a helpful error log, stops the worker if active, and raises & propagates the error
            '''
            LOGGER.error(f'Config attribute error: {err_msg}')
            try:
                cls.worker.stop()  # stop worker if active
            except AttributeError:
                pass
            raise ValidationError("Invalid config attribute.")
        
        def validate_attribute_type(
                attribute: str,
                expected_type: Literal['null_value', 'number_value', 'string_value', 'bool_value', 'struct_value', 'list_value'],
                ) -> None:
            '''
            validate_attribute_type exits and returns if the attribute doesn't exist,
            and handles the error for when the attribute does exist, but has an incorrect value type.
            '''
            value = attribute_map.get(key=attribute, default=None)
            if value is None:
                return  # user did not supply given attribute
            if value.WhichOneof('kind') != expected_type:
                handle_error(f'the "{attribute}" attribute must be a {expected_type}, not {value}.')

        def validate_dimension(attribute: str, max_value: int) -> None:
            '''
            validate_dimension helps validates height and width values.
            '''
            value = attribute_map.get(key=attribute, default=None)
            if value is None:
                return

            validate_attribute_type(attribute, 'number_value')
            number_value = value.number_value
            int_value = int(number_value)
            if int_value != number_value:
                handle_error(f'"{attribute}" must be a whole number.')
            if int_value > max_value:
                handle_error(f'inputted "{attribute}" of {int_value} cannot be greater than max of {max_value}.')
            if int_value <= 0:
                handle_error(f'inputted "{attribute}" of {int_value} cannot be less than or equal to 0.')
        
        # Check config keys are valid
        for attribute in attribute_map.keys():
            if attribute not in VALID_ATTRIBUTES:
                handle_error(f'"{attribute}" is not a valid attribute i.e. {VALID_ATTRIBUTES}')

        # Check debug is bool
        validate_attribute_type('debug', 'bool_value')

        # Check sensors is valid
        sensors_value = attribute_map.get(key='sensors', default=None)
        if sensors_value is None:
            handle_error('''
                        a "sensors" attribute of a list of sensor(s) to read data from is required
                        e.g. ["depth", "color"], with the first sensor in the list being the main sensor
                        that get_image uses.
                        ''')
        validate_attribute_type('sensors', 'list_value')
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
        if len(set(sensor_list)) != len(sensor_list):
            handle_error(f'please remove duplicates in the "sensors" attribute list: {sensor_list}')
                
        # Validate frame rate
        frame_rate = attribute_map.get(key='frame_rate', default=None)
        if frame_rate:
            if frame_rate.WhichOneof('kind') != 'number_value':
                handle_error(f'the "frame_rate" attribute must be a number value, not {frame_rate}.')
            if frame_rate.number_value > 60 or frame_rate.number_value <= 0:
                handle_error(f'"frame_rate" must be a number > 0 and <= 60.')

        # Check height value
        validate_dimension('height_px', MAX_HEIGHT)
        
        # Check width value
        validate_dimension('width_px', MAX_WIDTH)

        # Check height and width together
        height, width = attribute_map.get(key='height_px', default=None), attribute_map.get(key='width_px', default=None)
        if (height is None and width is not None) or (height is not None and width is None):
            handle_error('received only one dimension attribute. Please supply both "height_px" and "width_px", or neither.')

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        '''
        A procedure both the RDK and module invokes to (re)configure and (re)boot the module—
        serving as an initializer and restart method.

        Args:
            config (ComponentConfig)
            dependencies (Mapping[ResourceName, ResourceBase])
        '''        
        cls: OakDModel = type(self)
        try:
            LOGGER.debug('Trying to stop worker.')
            cls.worker.stop()
            LOGGER.info('Stopping active worker and reconfiguring...')
        except AttributeError:
            LOGGER.debug('No active worker.')

        self.camera_properties = Camera.Properties(
                supports_pcd=False,  # TODO: change to true after https://github.com/viamrobotics/viam-python-sdk/pull/481 is in
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

        user_wants_color, user_wants_depth = COLOR_SENSOR in self.sensors, DEPTH_SENSOR in self.sensors
        callback = lambda: self.reconfigure(config, dependencies)
        cls.worker = Worker(
            height=self.height,
            width=self.width,
            frame_rate=self.frame_rate,
            user_wants_color=user_wants_color,
            user_wants_depth=user_wants_depth,
            reconfigure=callback,
            logger=LOGGER
        )
        cls.worker.start()

    def stop(self, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> None:
        '''
        Implements ``stop`` under the Stoppable protocol to free resources.

        Args:
            extra (Optional[Mapping[str, Any]], optional): Unused.
            timeout (Optional[float], optional): Accepted. Defaults to None and will run with no timeout.
        '''
        cls: OakDModel = type(self)
        if timeout:
            self._run_with_timeout(timeout, cls.worker.stop)
        else:
            cls.worker.stop()

    async def get_image(
        self, mime_type: str = '', *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Union[Image.Image, RawImage]:
        '''
        Gets the next image from the camera as an Image or RawImage.
        Be sure to close the image when finished.

        NOTE: If the mime type is ``image/vnd.viam.dep`` you can use :func:`viam.media.video.RawImage.bytes_to_depth_array`
        to convert the data to a standard representation.

        Args:
            mime_type (str): The desired mime type of the image. This does not guarantee output type.

        Raises:
            NotSupportedError: if mime_type is not supported for the method
            ViamError: if validation somehow failed and no sensors are configured

        Returns:
            PIL.Image.Image | RawImage: The frame
        '''
        # LOGGER.debug('Handling get_image request.')
        mime_type =self._validate_get_image_mime_type(mime_type)
        cls: OakDModel = type(self)

        main_sensor = self.sensors[0]
        if main_sensor == COLOR_SENSOR:
            if mime_type == CameraMimeType.JPEG:
                captured_data = await cls.worker.get_color_image()
                return Image.fromarray(captured_data.np_array, 'RGB')
            raise NotSupportedError(f'mime_type "{mime_type}" is not supported for color. Please use {CameraMimeType.JPEG}')
        
        if main_sensor == DEPTH_SENSOR:
            captured_data = await cls.worker.get_depth_map()
            arr = captured_data.np_array
            if mime_type == CameraMimeType.JPEG:
                return Image.fromarray(arr, 'I;16').convert('RGB')
            if mime_type == CameraMimeType.VIAM_RAW_DEPTH:
                encoded_bytes = self._encode_depth_raw(arr.tobytes(), arr.shape)
                return RawImage(encoded_bytes, mime_type)
            raise NotSupportedError(f'mime_type {mime_type} is not supported for depth. Please use {CameraMimeType.JPEG} or {CameraMimeType.VIAM_RAW_DEPTH}.')
        raise ViamError('get_image failed due to misconfigured "sensors" attribute, but should have been validated in `validate`...')
    
    async def get_images(self, *, timeout: Optional[float] = None, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        '''
        Gets simultaneous images from different imagers, along with associated metadata.
        This should not be used for getting a time series of images from the same imager.

        Returns:
            Tuple[List[NamedImage], ResponseMetadata]:
                - List[NamedImage]:
                  The list of images returned from the camera system.

                - ResponseMetadata:
                  The metadata associated with this response
        '''
        cls: OakDModel = type(self)
        l: List[NamedImage] = []
        seconds_float: float = None
        if COLOR_SENSOR in self.sensors:
            captured_data = await cls.worker.get_color_image()
            arr, captured_at = captured_data.np_array, captured_data.captured_at

            # Create a Pillow image from the raw data
            pil_image = Image.fromarray(arr)

            # Create a BytesIO buffer to save the image as JPEG
            output_buffer = io.BytesIO()

            # Save the image as JPEG to the buffer
            pil_image.save(output_buffer, format='JPEG')

            # Get the bytes from the buffer
            jpeg_encoded_bytes = output_buffer.getvalue()
            img = NamedImage('color', jpeg_encoded_bytes, CameraMimeType.JPEG)
            seconds_float = captured_at
            l.append(img)
        if DEPTH_SENSOR in self.sensors:
            captured_data = await cls.worker.get_depth_map()
            arr, captured_at = captured_data.np_array, captured_data.captured_at
            depth_encoded_bytes = self._encode_depth_raw(arr.tobytes(), arr.shape)
            img = NamedImage('depth', depth_encoded_bytes, CameraMimeType.VIAM_RAW_DEPTH)
            seconds_float = captured_at
            l.append(img)
        seconds_int = int(seconds_float)
        nanoseconds_int = int((seconds_float - seconds_int) * 1e9)
        metadata = ResponseMetadata(captured_at=Timestamp(seconds=seconds_int, nanos=nanoseconds_int))
        return l, metadata
    
    async def get_point_cloud(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs
    ) -> Tuple[bytes, str]:
        '''
        Gets the next point cloud from the camera. This will be
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

        Raises:
            MethodNotAllowed: when config doesn't supply "depth" as a sensor

        Returns:
            bytes: The serialized point cloud data.
            str: The mimetype of the point cloud (e.g. PCD).
        '''
        if DEPTH_SENSOR not in self.sensors:
            details = 'Please include "depth" in the "sensors" attribute list.'
            raise MethodNotAllowed(method_name='get_point_cloud', details=details)
        cls = type(self)
        pcd_obj = await cls.worker.get_pcd()
        arr = pcd_obj.np_array
        # TODO RSDK-5676: why do we need to normalize by 1000 when depthAI says they return depth in mm?
        flat_array = arr.reshape(-1, arr.shape[-1]) / 1000.0
        # TODO RSDK-5676: why do we need to negate the 1st and 2nd dimensions for image to be the correct orientation?
        flat_array[:, 0:2] = -flat_array[:, 0:2]
        version = 'VERSION .7\n'
        fields = 'FIELDS x y z\n'
        size = 'SIZE 4 4 4\n'
        type_of = 'TYPE F F F\n'
        count = 'COUNT 1 1 1\n'
        height = 'HEIGHT 1\n'
        viewpoint = 'VIEWPOINT 0 0 0 1 0 0 0\n'
        data = 'DATA binary\n'
        width = f'WIDTH {len(flat_array)}\n'
        points = f'POINTS {len(flat_array)}\n'
        header = f'{version}{fields}{size}{type_of}{count}{width}{height}{viewpoint}{points}{data}'
        h = bytes(header, 'UTF-8')
        a = np.array(flat_array, dtype='f')
        return (h + a.tobytes(), CameraMimeType.PCD)

    
    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Properties:
        '''
        Gets the camera intrinsic parameters and camera distortion parameters

        Returns:
            Properties: The properties of the camera
        '''
        return self.camera_properties

    def _encode_depth_raw(self, data: bytes, shape: Tuple[int, int]) -> bytes:
        '''
        Encodes raw data into a bytes payload deserializable by the Viam SDK (camera mime type depth)

        Args:
            data (bytes): raw bytes
            shape (Tuple[int, int]): output dimensions of depth map (height x width)

        Returns:
            bytes: encoded bytes
        '''        
        height, width = shape  # using np array shape for actual outputted height/width
        MAGIC_NUMBER = struct.pack('>Q', 4919426490892632400)  # UTF-8 binary encoding for 'DEPTHMAP', big-endian
        MAGIC_BYTE_COUNT = struct.calcsize('Q')  # Number of bytes used to represent the depth magic number
        WIDTH_BYTE_COUNT = struct.calcsize('Q')  # Number of bytes used to represent depth image width
        HEIGHT_BYTE_COUNT = struct.calcsize('Q')  # Number of bytes used to represent depth image height

        # Depth header contains 8 bytes for the magic number, followed by 8 bytes for width and 8 bytes for height. Each pixel has 2 bytes.
        pixel_byte_count = np.dtype(np.uint16).itemsize * width * height
        width_to_encode = struct.pack('>Q', width)  # Q signifies big-endian
        height_to_encode = struct.pack('>Q', height)
        total_byte_count = MAGIC_BYTE_COUNT + WIDTH_BYTE_COUNT + HEIGHT_BYTE_COUNT + pixel_byte_count
        LOGGER.debug(f'{MAGIC_BYTE_COUNT} + {WIDTH_BYTE_COUNT} + {HEIGHT_BYTE_COUNT} + {pixel_byte_count} = total_byte_count: {total_byte_count}')
        LOGGER.debug(f'size of data: {len(data)}')

        # Create a bytearray to store the encoded data
        raw_buf = bytearray(total_byte_count)

        offset = 0

        # Copy the depth magic number into the buffer
        raw_buf[offset:offset + MAGIC_BYTE_COUNT] = MAGIC_NUMBER
        offset += MAGIC_BYTE_COUNT

        # Copy the encoded width and height into the buffer
        raw_buf[offset:offset + WIDTH_BYTE_COUNT] = width_to_encode
        offset += WIDTH_BYTE_COUNT
        raw_buf[offset:offset + HEIGHT_BYTE_COUNT] = height_to_encode
        offset += HEIGHT_BYTE_COUNT

        # Copy data into rest of the buffer
        raw_buf[offset:offset + pixel_byte_count] = data
        return bytes(raw_buf)
    
    def _validate_get_image_mime_type(self, mime_type: CameraMimeType) -> CameraMimeType:
        '''
        Validates inputted mime type for get_image calls.

        Args:
            mime_type (CameraMimeType): user inputted mime_type

        Raises:
            err: NotSupportedError

        Returns:
            CameraMimeType: validated/converted mime type
        '''        
        # Guard for empty str (no inputted mime_type)
        if mime_type == '':
            LOGGER.warn(f'mime_type was empty str or null; defaulting to {CameraMimeType.JPEG}.')
            return CameraMimeType.JPEG

        # Get valid types based on main sensor
        main_sensor = self.sensors[0]
        if main_sensor == 'color':
            valid_mime_types = [CameraMimeType.JPEG]
        else:  # depth
            valid_mime_types = [CameraMimeType.JPEG, CameraMimeType.VIAM_RAW_DEPTH]

        # Check validity
        if mime_type not in valid_mime_types:
            err = NotSupportedError(
                f'mime_type "{mime_type}" is not supported for get_image.'
                f'Valid mime type(s): {valid_mime_types}.'
            )
            LOGGER.error(err)
            raise err

        return mime_type
    
    def _run_with_timeout(self, timeout: float, function: Callable) -> None:
        '''
        Run a function with timeout.

        Args:
            timeout (float)
            function (Callable)
        '''        
        thread = threading.Thread(target=function)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            thread.join()
            LOGGER.error(f'{function.__name__} timed out after {timeout} seconds.')


class MethodNotAllowed(ViamError):
    '''
    Exception raised when attempting to call a method
    with a configuration that does not support said method.
    '''
    def __init__(self, method_name: str, details: str) -> None:
        self.name = method_name
        self.message = f'Cannot invoke method "{method_name}" with current config. {details}'
        super().__init__(self.message)
