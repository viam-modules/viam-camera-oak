# OAK-D Modular Component

This is a [Viam module](https://docs.viam.com/manage/configuration/#modules) for the [OAK-D](https://shop.luxonis.com/products/oak-d) camera. Registered at https://app.viam.com/module/viam/oak-d.

## Getting Started

### Checking Python version

First and foremost, open a terminal on your robot, and run the following command to check its Python version:

```console
$ python3 --version
```

Verify that your robot has Python 3.8 or a later version installed and running to avoid compatibility issues.

### Using the registry

The recommended way to install the module is through the Viam registry.

- Go to your robot's page on app.viam.com.
- Click on the *Create Component* button in the Components section.
- Search for the *oak-d* module and select it. 

This will automatically install the module to your robot.


### Locally installing the module

If you do not want to use the Viam registry, you can use the module from source [here.](https://github.com/viamrobotics/viam-camera-oak-d)

```console
$ cd <your-directory>
$ git clone https://github.com/viamrobotics/viam-camera-oak-d.git
```

Then modify your robot's JSON file as follows

```
  "modules": [
    {
      "type": "local",
      "name": "oak-d",
      "executable_path": "<your-directory>/viam-camera-oak-d/run.sh"
    }
  ],
```

## Attributes and Sample Config

The attributes for the module are as follows:
- `sensors` (required): a list that contain the strings `color` and/or `depth`. The sensor that comes first in the list is designated the "main sensor" and will be the image that gets returned by `get_image` calls and what will appear in the Control tab on app.viam.
- `width_px`, `height_px`: the int width and height of the output images. If the OAK-D cannot produce the requested resolution, the component will be configured to the closest resolution to the given height/width. Therefore, the image output size will not always match the input size.
- `frame_rate`: the float that represents the frame rate the camera will capture images at.
- `debug`: the bool that determines whether the module will log debug messages in `std.out`.
```
{
  "components": [
    {
      "name": "my-oak-d-camera",
      "attributes": {
        "sensors": ["color", "depth"],
        "width_px": 640,
        "height_px": 480,
        "frame_rate": 30,
        "debug": false
      },
      "namespace": "rdk",
      "type": "camera",
      "model": "viam:camera:oak-d"
    }
  ]
}
```
