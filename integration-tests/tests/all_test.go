package tests

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/edaniels/golog"
	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/config"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/robot"
	robotimpl "go.viam.com/rdk/robot/impl"
	"go.viam.com/rdk/utils"
	"go.viam.com/test"
)

const (
	componentName string = "my-oak-d"
	// Default values should mirror those at the top of oak_d.py
	defaultWidth  int = 640
	defaultHeight int = 400
)

func TestCameraServer(t *testing.T) {
	var myRobot robot.Robot
	var cam camera.Camera
	t.Run("Set up the robot", func(t *testing.T) {
		var err error
		myRobot, err = setUpViamServer(context.Background(), t)
		test.That(t, err, test.ShouldBeNil)
		cam, err = camera.FromRobot(myRobot, componentName)
		test.That(t, err, test.ShouldBeNil)
	})

	t.Run("Get images method", func(t *testing.T) {
		images, metadata, err := cam.Images(context.Background())
		test.That(t, err, test.ShouldBeNil)
		test.That(t, images, test.ShouldNotBeNil)
		test.That(t, metadata, test.ShouldNotBeNil)
		for _, img := range images {
			test.That(t, img.SourceName, test.ShouldEqual, componentName)
			bounds := img.Image.Bounds()
			test.That(t, bounds.Dx(), test.ShouldEqual, defaultWidth)
			test.That(t, bounds.Dy(), test.ShouldEqual, defaultHeight)
		}
		test.That(t, metadata.CapturedAt, test.ShouldHappenBefore, time.Now())
	})

	t.Run("Get point cloud method", func(t *testing.T) {
		pc, err := cam.NextPointCloud(context.Background())
		test.That(t, err, test.ShouldBeNil)
		test.That(t, pc, test.ShouldNotBeNil)
		test.That(t, pc.Size(), test.ShouldBeBetweenOrEqual, defaultHeight*defaultWidth-100, defaultHeight*defaultWidth)
	})

	t.Run("Reconfigure module", func(t *testing.T) {
		cfg := resource.Config{
			Attributes: utils.AttributeMap{
				"sensors": []string{"color"},
			},
		}
		err := cam.Reconfigure(context.Background(), resource.Dependencies{}, cfg)
		test.That(t, err, test.ShouldBeNil)
	})

	t.Run("Shut down the camera", func(t *testing.T) {
		test.That(t, cam.Close(context.Background()), test.ShouldBeNil)
	})

	t.Run("Shut down the robot", func(t *testing.T) {
		test.That(t, myRobot.Close(context.Background()), test.ShouldBeNil)
	})
}

func setUpViamServer(ctx context.Context, t *testing.T) (robot.Robot, error) {
	logger := golog.NewTestLogger(t)
	moduleString := strings.TrimSpace(*modulePath)
	logger.Info("testing module at %v", moduleString)
	configString := fmt.Sprintf(`
		{
		"network": {
			"bind_address": "0.0.0.0:90831",
			"insecure": true
		},
		"components": [
			{
			"name": "%v",
			"model": "viam:camera:oak-d",
			"type": "camera",
			"namespace": "rdk",
			"attributes": {
				"sensors": [
				"color",
				"depth"
				]
			},
			"depends_on": []
			}
		],
		"modules": [
			{
			"type": "local",
			"name": "viam_oak_d",
			"executable_path": "%v"
			}
		]
		}
	`, componentName, moduleString)

	cfg, err := config.FromReader(ctx, "default.json", bytes.NewReader([]byte(configString)), logger)
	if err != nil {
		return nil, err
	}

	r, err := robotimpl.RobotFromConfig(ctx, cfg, logger)
	if err != nil {
		return nil, err
	}

	return r, nil
}
