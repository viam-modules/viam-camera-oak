# OAK-D Modular Component

This is a [Viam module](https://docs.viam.com/manage/configuration/#modules) for the [OAK-D](https://shop.luxonis.com/products/oak-d) camera. Registered at https://app.viam.com/module/viam/oak-d.

## Getting Started

### Checking Python version (for local installs and non-AArch64 robots)

If you installed the module locally or if your robot is using a non-AArch64 board, you must verify that your system Python3 is compatible with Viam. Open a terminal on your robot, and run the following commands to check its Python and pip versions:

```console
sudo python3 --version
sudo python3 -c "import venv"
sudo python3 -m ensurepip --default-pip
sudo pip3 --version
```

Verify that your robot's Python3 version is 3.8.1 or later, and that it is installed and linked to the `python3` command to avoid compatibility issues.
Similarly, make sure that `venv` and `pip3` are installed properly by making sure the subsequent commands do not produce an error.

If you are using the registry to install the module and your robot is using an AArch64 board such as a 64-bit Raspberry Pi or a Jetson device, ignore these instructions as the module will be bundled as an Appimage, which includes Python and necessary dependencies statically.

### Using the registry

The recommended way to install the module is through the Viam registry.

- Go to your robot's page on app.viam.com.
- Click on the *Create Component* button in the Components section.
- Search for the *oak-d* module and select it. 

This will automatically install the module to your robot.

### Locally installing the module

If you do not want to use the Viam registry, you can use the module from source [here](https://github.com/viamrobotics/viam-camera-oak-d).

```console
cd <path-to-your-directory>
git clone https://github.com/viamrobotics/viam-camera-oak-d.git
```

Then modify your robot's JSON file as follows

```
  "modules": [
    {
      "type": "local",
      "name": "oak-d",
      "executable_path": "<path-to-your-directory>/viam-camera-oak-d/packaging/run.sh"
    }
  ],
```

## Attributes and Sample Config

The attributes for the module are as follows:
- `sensors` (required): an array that contains the strings `color` and/or `depth`. The sensor that comes first in the array is designated the "main sensor" and will be the image that gets returned by `get_image` calls and what will appear in the Control tab on app.viam. When both sensors are requested, `get_point_clouds` will be available for use, and `get_images` will return both the color and depth outputs. Additionally, color and depth outputs returned together will always be aligned, have the same height and width, and have the same timestamp. See Viam's [documentation on the Camera API](https://docs.viam.com/components/camera/#api) for more details. 
- `width_px`, `height_px`: the int width and height of the output images. If the OAK-D cannot produce the requested resolution, the component will be configured to the closest resolution to the given height/width. Therefore, the image output size will not always match the input size. `width_px` defaults to `640` and `height_px` defaults to `400`. Note: higher resolutions may cause out of memory errors. See Luxonis documentation [here](https://docs.luxonis.com/projects/api/en/latest/tutorials/ram_usage/).
- `frame_rate`: the float that represents the frame rate the camera will capture images at. Defaults to `30.0`.

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
      },
      "namespace": "rdk",
      "type": "camera",
      "model": "viam:camera:oak-d"
    }
  ]
}
```

Although not a config attribute, you can also configure the module to output debug logs. This is done by using the `-debug` flag 
when starting the Viam server in order for module debug logs to be piped through to stdout e.g. `viam-server -debug -config path/to/your/config.json`.

## Integration Tests

The repo comes with a suite of integration tests that allows one to test if the module works with an actual OAK-D device on the machine of interest. You will need to compile the binary on the same machine you expect to run it on.

- Copy the repo to your local robot: `git clone https://github.com/viamrobotics/viam-camera-oak-d.git`
- Run `make integration-tests`
