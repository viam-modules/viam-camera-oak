# OAK Modular Component

This is a [Viam module](https://docs.viam.com/manage/configuration/#modules) for the [OAK](https://shop.luxonis.com/collections/oak-cameras-1) family of cameras. Registered at https://app.viam.com/module/viam/oak.

## Build and Run

### Configure your camera

> [!NOTE]  
> Before configuring your camera, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Components** subtab and click **Create component**.
Select the `camera` type, then select the `oak` model.
Enter a name for your camera and click **Create**.

On the new component panel, copy and paste the following attribute template into your camera’s **Attributes** box:

```json
{
  "sensors": ["color", "depth"],
  "width_px": 640,
  "height_px": 480,
  "frame_rate": 30,
}
```

Edit these attributes as applicable to your machine. 

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

## Attributes

The following attributes are available for `oak` cameras:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `sensors` | array | **Required** | An array that contains the strings `color` and/or `depth`. The sensor that comes first in the array is designated the "main sensor" and will be the image that gets returned by `get_image` calls and what will appear in the Control tab on the [Viam app](https://app.viam.com) When both sensors are requested, `get_point_clouds` will be available for use, and `get_images` will return both the color and depth outputs. Additionally, color and depth outputs returned together will always be aligned, have the same height and width, and have the same timestamp. See Viam's [documentation on the Camera API](https://docs.viam.com/components/camera/#api) for more details.  |
| `width_px` | int | Optional | Width in pixels of the images output by this camera. If the camera cannot produce the requested resolution, the component will be configured to the closest resolution to the given height/width. Therefore, the image output size will not always match the input size. Default: `1280` |
| `height_px` | int | Optional | Height in pixels of the images output by this camera. If the camera cannot produce the requested resolution, the component will be configured to the closest resolution to the given height/width. Therefore, the image output size will not always match the input size. Default: `720` |
| `frame_rate` | int | Optional | The frame rate the camera will capture images at. Default: `30` |

> [!NOTE]  
> Higher resolutions may cause out of memory errors. See Luxonis documentation [here](https://docs.luxonis.com/projects/api/en/latest/tutorials/ram_usage/.).

### Example Configuration

```
{
  "components": [
    {
      "name": "my-oak-camera",
      "attributes": {
        "sensors": ["color", "depth"],
        "width_px": 640,
        "height_px": 480,
        "frame_rate": 30,
      },
      "namespace": "rdk",
      "type": "camera",
      "model": "viam:camera:oak"
    }
  ]
}
```

### Debugging

Although not a config attribute, you can also configure the module to output debug logs.
This is done by using the `-debug` flag when starting the Viam server in order for module debug logs to be piped through to stdout e.g. `viam-server -debug -config path/to/your/config.json`.

### Set udev rules on Linux

- Failed to boot the device: 1.3-ma2480, err code 3
- Failed to find device (ma2480), error message: X_LINK_DEVICE_NOT_FOUND
- [warning] skipping X_LINK_UNBOOTED device having name "<error>"
- Insufficient permissions to communicate with X_LINK_UNBOOTED device with name "1.1". Make sure udev rules are set

If you see any of the above errors, you may need to set udev rules on your Linux machine. See [here](https://docs.luxonis.com/en/latest/pages/troubleshooting/?highlight=udev#udev-rules-on-linux) for more information.

```console
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Locally installing the module

If you do not want to use the Viam registry, you can use the module from source [here](https://github.com/viamrobotics/viam-camera-oak).

You must, however, verify that your system Python3 is compatible with Viam to run the module locally. Open a terminal on your robot, and run the following commands to check its Python and pip versions:

```console
sudo python3 --version
sudo python3 -c "import venv"
sudo python3 -m ensurepip --default-pip
sudo pip3 --version
```

Verify that your robot's Python3 version is 3.8.1 or later, and that it is installed and linked to the `python3` command to avoid compatibility issues.
Similarly, make sure that `venv` and `pip3` are installed properly by making sure the subsequent commands do not produce an error.

```console
cd <path-to-your-directory>
git clone https://github.com/viamrobotics/viam-camera-oak.git
```

Then modify your robot's JSON file as follows

```
  "modules": [
    {
      "type": "local",
      "name": "oak",
      "executable_path": "<path-to-your-directory>/viam-camera-oak/local_run.sh"
    }
  ],
```

## Integration Tests

The repo comes with a suite of integration tests that allows one to test if the module works with an actual OAK device on the machine of interest. You will need to compile the binary on the same machine you expect to run it on.

- Copy the repo to your local robot: `git clone https://github.com/viamrobotics/viam-camera-oak.git`
- Run `make integration-tests`
