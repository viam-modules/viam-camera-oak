package tests

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/config"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/robot"
	robotimpl "go.viam.com/rdk/robot/impl"
	"go.viam.com/rdk/utils"
	"go.viam.com/test"
)

const (
	componentName string = "oak-d-cam"
	// Default values should mirror those in source code
	defaultWidth            int = 1280
	defaultHeight           int = 720
	maxGRPCMessageByteCount     = 4194304 // Update this if the gRPC config ever changes
)

func TestCameraServer(t *testing.T) {
	timeoutCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	var myRobot robot.Robot
	var cam camera.Camera
	t.Run("Set up the robot", func(t *testing.T) {
		var err error
		myRobot, err = setUpViamServer(timeoutCtx, t)
		test.That(t, err, test.ShouldBeNil)
		cam, err = camera.FromRobot(myRobot, componentName)
		test.That(t, err, test.ShouldBeNil)
	})

	t.Run("Get images method", func(t *testing.T) {
		images, metadata, err := cam.Images(timeoutCtx)
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
		pc, err := cam.NextPointCloud(timeoutCtx)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, pc, test.ShouldNotBeNil)
		test.That(t, pc.Size(), test.ShouldBeBetween, 0, maxGRPCMessageByteCount)
	})

	t.Run("Reconfigure module", func(t *testing.T) {
		cfg := resource.Config{
			Attributes: utils.AttributeMap{
				"sensors": []string{"color"},
			},
		}
		err := cam.Reconfigure(timeoutCtx, resource.Dependencies{}, cfg)
		test.That(t, err, test.ShouldBeNil)
	})

	t.Run("Shut down the camera", func(t *testing.T) {
		test.That(t, cam.Close(timeoutCtx), test.ShouldBeNil)
	})

	t.Run("Shut down the robot", func(t *testing.T) {
		test.That(t, myRobot.Close(timeoutCtx), test.ShouldBeNil)
	})
}

func setUpViamServer(ctx context.Context, _ *testing.T) (robot.Robot, error) {
	logger := logging.NewLogger("oak-integration-tests-logger")

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
			"model": "viam:luxonis:oak-d",
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
			"name": "viam_oak",
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
