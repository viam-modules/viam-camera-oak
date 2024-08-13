package tests

import (
	"context"
	"fmt"
	"testing"
	"time"

	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/robot"
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
	timeoutCtx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	var myRobot robot.Robot
	var cam camera.Camera
	t.Run("Set up the robot", func(t *testing.T) {
		var err error
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
		`, componentName, absModulePath)
		myRobot, err = setUpViamServer(timeoutCtx, configString, "oak_d_test", t)
		test.That(t, err, test.ShouldBeNil)
		cam, err = camera.FromRobot(myRobot, componentName)
		test.That(t, err, test.ShouldBeNil)
	})

	t.Run("Get images method (two images)", func(t *testing.T) {
		images, metadata, err := cam.Images(timeoutCtx)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, images, test.ShouldNotBeNil)
		test.That(t, metadata, test.ShouldNotBeNil)
		test.That(t, len(images), test.ShouldEqual, 2)
		for _, img := range images {
			test.That(t, img.SourceName, test.ShouldEqual, componentName)
			bounds := img.Image.Bounds()
			test.That(t, bounds.Dx(), test.ShouldEqual, defaultWidth)
			test.That(t, bounds.Dy(), test.ShouldEqual, defaultHeight)
		}
		test.That(t, metadata.CapturedAt, test.ShouldHappenBefore, time.Now())
	})

	t.Run("Reconfigure camera", func(t *testing.T) {
		cfg := resource.Config{
			Attributes: utils.AttributeMap{
				"sensors": []string{"depth"},
			},
		}
		err := cam.Reconfigure(timeoutCtx, resource.Dependencies{}, cfg)
		test.That(t, err, test.ShouldBeNil)
	})

	t.Run("Get point cloud method", func(t *testing.T) {
		pc, err := cam.NextPointCloud(timeoutCtx)
		test.That(t, err, test.ShouldBeNil)
		test.That(t, pc, test.ShouldNotBeNil)
		test.That(t, pc.Size(), test.ShouldBeBetween, 0, maxGRPCMessageByteCount)
	})

	t.Run("Shut down the camera", func(t *testing.T) {
		test.That(t, cam.Close(timeoutCtx), test.ShouldBeNil)
	})

	t.Run("Shut down the robot", func(t *testing.T) {
		test.That(t, myRobot.Close(timeoutCtx), test.ShouldBeNil)
	})
}

