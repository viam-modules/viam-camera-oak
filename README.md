# [`oak` module](https://app.viam.com/module/viam/oak)

This is a [Viam module](https://docs.viam.com/manage/configuration/#modules) that implements the [`rdk:component:camera` API](https://docs.viam.com/components/camera/#api) in two models and [`rdk:service:vision` API](https://docs.viam.com/services/vision/#api) in one model for the [OAK](https://shop.luxonis.com/collections/oak-cameras-1) family of cameras. 
The module supports:
- getting color frames, stereo depth frames, and point cloud data from OAK cameras using the `oak-d` and `oak-ffc-3p` component models
- setting up a YOLO detection network on the camera VPU using the `yolo-detection-network` vision service model.

> [!NOTE]  
> Learn about the difference between modules, models, components, and services in [our docs](https://docs.viam.com/appendix/glossary/).

## Requirements
The module executable is currently only supported on `linux/arm64`, `linux/amd64`, and `darwin/arm64`. Make sure your machine is running on one of these architectures to avoid exec format issues.

## Supported Models

### Components
* `viam:luxonis:oak-d` - Configures the [OAK-D](https://shop.luxonis.com/products/oak-d) integrated camera model.
* `viam:luxonis:oak-ffc-3p` - Configures the [OAK-FFC-3P](https://shop.luxonis.com/products/oak-ffc-3p) non-integrated model with flexible board and camera sensor support.

Other OAK family cameras may work with the either component model, but are not officially supported and tested.

### Services
* `viam:luxonis:yolo-detection-network` - Configures [a DepthAI Yolo Detection Network](https://docs.luxonis.com/software/depthai-components/nodes/yolo_detection_network/) on the OAK device pipeline.

## How to use on your own machine
> [!NOTE]  
> Before configuring your resource, you must create a new smart machine. See [here](https://docs.viam.com/use-cases/configure/#configure-a-machine) for how to do that and more on the setup and configuration process as a whole.

Navigate to the **CONFIGURE** tab of your machine's page in [the Viam app](https://app.viam.com/).

[Add `camera` / `oak-d` or `camera` / `oak-ffc-3p` to your machine](https://docs.viam.com/configure/#components).
Enter a name for your camera and click **Create**.

[Add `vision` service `yolo-detection-network` to your machine](https://docs.viam.com/configure/#services).
Enter a name for your service and click **Create**.

On the new resource config panel, modify the attributes JSON in the **Attributes** box. If you are confused which attributes to supply, consult the per-model attribute configuration guides below.

## Attribute configuration guides

### `oak-d` component model

Copy and paste the following attributes into your JSON configuration:

```json
{
    "sensors": ["color", "depth"],
}
```

#### Attributes

The following attributes are available for the `oak-d` camera component:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `sensors` | array | **Required** | An array that contains the strings `color` and/or `depth`. The sensor that comes first in the array is designated the "main sensor" and will be the image that gets returned by `get_image` calls and what will appear in the Control tab on the [Viam app](https://app.viam.com) When both sensors are requested, `get_point_clouds` will be available for use, and `get_images` will return both the color and depth outputs. Additionally, color and depth outputs returned together will always be aligned, have the same height and width, and have the same timestamp. See Viam's [documentation on the Camera API](https://docs.viam.com/components/camera/#api) for more details.  |
| `width_px` | int | Optional | Width in pixels of the images output by this camera. Default: `1280` |
| `height_px` | int | Optional | Height in pixels of the images output by this camera. Default: `720` |
| `frame_rate` | int | Optional | The frame rate the camera will capture images at. Default: `30` |
| `device_info` | string | Optional | Physical device identifier to connect to a specific OAK camera connected to your machine. If not specified, the module will pick the first device it detects. `device_info` can be a MXID, usb port path, or IP address. [See DepthAI documentation for more details](https://docs.luxonis.com/software/depthai/examples/device_information#Device%20information). |
| `manual_focus` | int | Optional | The manual focus value to apply to the color sensor. Sets the camera to fixed focus mode at the specified lens position. Must be between 0..255 inclusive. Default: auto focus |
| `right_handed_system` | bool | Optional | The OAK outputs point clouds in a left-handed coordinate system by default. Viam's frame system uses right-handed coordinates. By setting to `true` all outputted PCD will have their y values negated to be compatible with right-handed systems. Default: `false` |
| `point_cloud_enabled` | bool | Optional | Whether or not to enable point cloud functionality. Note that this is CPU intensive and should only be used when PCD is necessary. Default: `false` |

Note that higher resolutions may cause out of memory errors. See Luxonis documentation [here](https://docs.luxonis.com/projects/api/en/latest/tutorials/ram_usage/.).

#### Example Full Configuration
Copy and paste the following attributes into your camera's JSON configuration:
```json
{
    "sensors": ["color", "depth"],
    "width_px": 416,
    "height_px": 416,
    "frame_rate": 30,
    "right_handed_system": true,
    "point_cloud_enabled": false,
    "device_info": "<mxid-or-ip-address-or-usb-port-name>"
}
```

### `oak-ffc-3p` component model
Copy and paste the following attributes into your JSON configuration:
```json
{
  "camera_sensors": [
      {
          "socket": "cam_a",
          "type": "color",
      },
      {
          "socket": "cam_b",
          "type": "depth",
      },
      {
          "socket": "cam_c",
          "type": "color",
      }
  ]
}
```

#### Attributes

The following attributes are available for the `oak-ffc-3p` component:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera_sensors` | list[struct] | **Required** | A list of struct mappings of strings to values representing the sub-configuration per camera sensor on the device. The first element of this list will be considered the primary sensor. |
| `device_info` | string | Optional | Physical device identifier to connect to a specific OAK camera connected to your machine. If not specified, the module will pick the first device it detects. `device_info` can be a MXID, usb port path, or IP address. [See DepthAI documentation for more details](https://docs.luxonis.com/software/depthai/examples/device_information#Device%20information). |

The below attributes are nested inside each camera sensor struct inside `camera_sensors`:
| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `type` | str | **Required** | The type of the sensor: "color" or "depth", corresponding to whether the sensor on the respective socket is a color or mono depth camera. |
| `socket` | str | **Required** | The socket the sensor is connected to: "cam_a", "cam_b", or "cam_c", corresponding to the three available sensor sockets on the OAK-FFC-3P. Read more about DepthAI camera sockets on [their docs](https://docs.luxonis.com/software/api/python/#depthai.CameraBoardSocket). |
| `width_px` | int | Optional | Width in pixels of the images output by this camera. Default: `1280` |
| `height_px` | int | Optional | Height in pixels of the images output by this camera. Default: `720` |
| `frame_rate` | int | Optional | The frame rate the camera will capture images at. Default: `30` |
| `color_order` | str | Optional | The color order of the output frames (used for color sensor type only): "rgb" or "bgr". Default: `rgb` |
| `interleaved` | bool | Optional | Whether or not output frames should be stored in an interleaved format. Default: `false` |

Note that higher resolutions may cause out of memory errors. See Luxonis documentation [here](https://docs.luxonis.com/projects/api/en/latest/tutorials/ram_usage/.).

Below is an example of a full JSON of an `oak-ffc-3p` component:
```json
{
  "device_info": "<mxid-or-ip-address-or-usb-port-name>",
  "camera_sensors": [
      {
          "socket": "cam_a",
          "type": "color",
          "width_px": 416,
          "height_px": 416,
          "frame_rate": 30,
          "color_order": "rgb",
          "interleaved": false
      },
      {
          "socket": "cam_b",
          "type": "color",
          "width_px": 416,
          "height_px": 416,
          "frame_rate": 30,
          "color_order": "rgb",
          "interleaved": false
      },
      {
          "socket": "cam_c",
          "type": "color",
          "width_px": 416,
          "height_px": 416,
          "frame_rate": 30,
          "color_order": "rgb",
          "interleaved": false
      }
  ]
}
```

### `yolo-detection-network` service model

Copy and paste the following attributes into your JSON configuration:
```json
{
  "cam_name": "my-oak",
  "input_source": "cam_a",
  "yolo_config": {
    "blob_path": "/path/to/a/model/blob/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob",
    "labels": [
      "person", "bicycle", "car", "motorbike"
    ],
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5,
  }
}
```

#### Attributes

The following attributes are available for the `yolo-detection-network` service:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `cam_name` | string | **Required** | The name of the Viam OAK camera component to set up the service on. |
| `input_source` | string | **Required** | The socket name i.e. "cam_a", "cam_b", "cam_c" to get neural network input frames from. Can also be set as "color" to select the primary color sensor on the underlying OAK component. |
| `yolo_config` | struct | **Required** | A struct mapping of strings to values representing the sub-configuration for the YOLO model. |
| `num_nce_per_thread` | int | Optional | How many Neural Compute Engines should a single thread use for inference. Default `1` |
| `num_threads` | int | Optional | How many threads should the DepthAI node use to run the network. Can be 0, 1, or 2, where 0 is auto. Default `1` |

The below attributes are nested inside the `yolo_config` struct:
| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `blob_path` | string | **Required** | The local path to the YOLO model blob. The model must be in the .blob format compatible with Luxonis VPUs. See [here](https://docs.luxonis.com/software/ai-inference/conversion/) for more information. You can find many .blob formatted models to use in the [DepthAI Model Zoo](https://blobconverter.luxonis.com/), or use the converter to make your own. |
| `labels` | list[string] | **Required** | A list of strings representing each label the YOLO model can output. Order matters here as YOLO models use label index mapping to retrieve string labels. |
| `confidence_threshold` | float | **Required** | The minimum confidence level required for a detection to be considered valid. Default `0.5`. |
| `iou_threshold` | float | **Required** | The Intersection Over Union (IOU) threshold used for non-max suppression to filter out overlapping bounding boxes. Default `0.5`. |
| `anchors` | list[float] | Optional | A list of floats representing the anchor boxes used by the YOLO model. The values should be in pairs representing width and height. Default: `[]` |
| `anchor_masks` | dict | Optional | A dictionary where keys are strings (e.g., "side26", "side13") representing different scales, and values are lists of integers representing the anchor indices used at each scale. Default: `{}` |
| `coordinate_size` | int | Optional | The number of coordinates used for each bounding box. Typically, this is 4 for (x, y, width, height). Default `4`. |

Below is an example full configuration of a `yolo-detection-network` service:
```json
{
  "cam_name": "my-oak",
  "input_source": "cam_a",
  "num_threads": 2,
  "num_nce_per_thread": 1,
  "yolo_config": {
    "blob_path": "/path/to/a/model/blob/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob",
    "labels": [
      "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ],
    "confidence_threshold": 0.5,
    "iou_threshold": 0.5,
    "anchors": [
      10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319
    ],
    "anchor_masks": {
      "side13": [
        3, 4, 5
      ],
      "side26": [
        1, 2, 3
      ]
    },
    "coordinate_size": 4,
  }
}
```


## Next steps

### View camera outputs in App

Once your machine is properly configured and started, go to your machine in the Viam App, find your resource's resource card in the **CONFIGURE** tab and click **TEST**. This will display the output frames of your camera component or bounding box detections image from your vision service.

To see point cloud outputs, go to **CONTROL**, scroll to your camera component, and click **View point cloud data**.

### Debugging

Although not a config attribute, you can also configure the module to output debug logs.
This is done by using the `-debug` flag when starting the Viam server in order for module debug logs to be piped through to stdout e.g. `viam-server -debug -config path/to/your/config.json`.

### Set udev rules on Linux

- Failed to boot the device: 1.3-ma2480, err code 3
- Failed to find device (ma2480), error message: X_LINK_DEVICE_NOT_FOUND
- [warning] skipping X_LINK_UNBOOTED device having name "<error>"
- Insufficient permissions to communicate with X_LINK_UNBOOTED device with name "1.1". Make sure udev rules are set

If you see any of the above errors, you may need to set udev rules on your Linux machine. See [here](https://github.com/luxonis/depthai-docs-website/blob/master/source/pages/troubleshooting.rst) for more information.

```console
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Locally installing the module

If you do not want to use the Viam registry, you can use the module from source [here](https://github.com/viam-modules/viam-camera-oak).

You must, however, verify that your system Python3 is compatible with Viam to run the module locally. Open a terminal on your machine, and run the following commands to check its Python and pip versions:

```console
sudo python3 --version
sudo python3 -c "import venv"
sudo python3 -m ensurepip --default-pip
sudo pip3 --version
```

Verify that your machine's Python3 version is 3.8.1 or later, and that it is installed and linked to the `python3` command to avoid compatibility issues.
Similarly, make sure that `venv` and `pip3` are installed properly by making sure the subsequent commands do not produce an error.

```console
cd <path-to-your-directory>
git clone https://github.com/viam-modules/viam-camera-oak.git
```

Then modify your machine's config file as per the above instructions per model, and start the machine.

### Integration Tests

The repo comes with a suite of integration tests that allows one to test if the module works with an actual OAK device on the machine of interest. You will need to compile the binary on the same machine you expect to run it on.

- Copy the repo to your local machine: `git clone https://github.com/viam-modules/viam-camera-oak.git`
- Run `make integration-tests`
