package tests

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"go.viam.com/rdk/components/camera"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/robot"
	"go.viam.com/rdk/services/vision"
	"go.viam.com/rdk/utils"
	"go.viam.com/rdk/vision/objectdetection"
	"go.viam.com/rdk/vision/viscapture"
	"go.viam.com/test"
)

const (
	serviceName   string = "ydn"
	defaultYdnWidth  int    = 416
	defaultYdnHeight int    = 416
)

func TestYoloDetectionNetwork(t *testing.T) {
	if absBlobPath == "" {
		return
	}
	timeoutCtx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	var myRobot robot.Robot
	var cam camera.Camera
	var ydn vision.Service
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
					],
					"width_px": %d,
					"height_px": %d
				},
				"depends_on": []
				}
			],
			"services": [
				{
				"name": "%v",
				"namespace": "rdk",
				"type": "vision",
				"model": "viam:luxonis:yolo-detection-network",
				"attributes": {
					"cam_name": "%v",
					"input_source": "color",
					"num_threads": 2,
					"num_nce_per_thread": 1,
					"yolo_config": {
						"iou_threshold": 0.5,
						"coordinate_size": 4,
						"blob_path": "%v",
						"labels": [
							"person",
							"bicycle",
							"car",
							"motorbike",
							"aeroplane",
							"bus",
							"train",
							"truck",
							"boat",
							"traffic light",
							"fire hydrant",
							"stop sign",
							"parking meter",
							"bench",
							"bird",
							"cat",
							"dog",
							"horse",
							"sheep",
							"cow",
							"elephant",
							"bear",
							"zebra",
							"giraffe",
							"backpack",
							"umbrella",
							"handbag",
							"tie",
							"suitcase",
							"frisbee",
							"skis",
							"snowboard",
							"sports ball",
							"kite",
							"baseball bat",
							"baseball glove",
							"skateboard",
							"surfboard",
							"tennis racket",
							"bottle",
							"wine glass",
							"cup",
							"fork",
							"knife",
							"spoon",
							"bowl",
							"banana",
							"apple",
							"sandwich",
							"orange",
							"broccoli",
							"carrot",
							"hot dog",
							"pizza",
							"donut",
							"cake",
							"chair",
							"sofa",
							"pottedplant",
							"bed",
							"diningtable",
							"toilet",
							"tvmonitor",
							"laptop",
							"mouse",
							"remote",
							"keyboard",
							"cell phone",
							"microwave",
							"oven",
							"toaster",
							"sink",
							"refrigerator",
							"book",
							"clock",
							"vase",
							"scissors",
							"teddy bear",
							"hair drier",
							"toothbrush"
						],
						"confidence_threshold": 0.5
					}
				}
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
		`, componentName, defaultYdnWidth, defaultYdnHeight, serviceName, componentName, absBlobPath, absModulePath)
		myRobot, err = setUpViamServer(timeoutCtx, configString, "ydn_test", t)
		test.That(t, err, test.ShouldBeNil)
		cam, err = camera.FromRobot(myRobot, componentName)
		ydn, err = vision.FromRobot(myRobot, serviceName)
		test.That(t, err, test.ShouldBeNil)
		time.Sleep(5*time.Second)  // wait a bit for do_command to do its thing
	})

	t.Run("Get detections from camera", func(t *testing.T) {
		attempts := 0
		var dets []objectdetection.Detection
		var err error

		for attempts < 3 {
			dets, err = ydn.DetectionsFromCamera(timeoutCtx, "", nil)
			if err != nil {
				errStr := err.Error()
				if strings.Contains(errStr, "Camera data requested before camera worker was ready") || strings.Contains(errStr, "Could not find matching YDN config for YDN service id") {
					attempts++
					time.Sleep(1*time.Second)
					continue
				} else {
					t.Errorf("Unexpected error occurred getting detections from camera: %v", errStr)
					break
				}
			}
			break
		}

		test.That(t, dets, test.ShouldNotBeNil)
		for _, det := range dets {
			test.That(t, det.BoundingBox().Max.X, test.ShouldBeLessThanOrEqualTo, defaultYdnWidth)
			test.That(t, det.BoundingBox().Max.Y, test.ShouldBeLessThanOrEqualTo, defaultYdnHeight)
			test.That(t, det.Label(), test.ShouldNotBeNil)
			test.That(t, det.Label(), test.ShouldNotBeEmpty)
			test.That(t, det.Score(), test.ShouldBeLessThanOrEqualTo, 1)
		}
	})


	t.Run("Reconfigure service", func(t *testing.T) {
		cfg := resource.Config{
			Attributes: utils.AttributeMap{
				"cam_name":            componentName,
				"input_source":        "cam_a",
				"num_threads":         2,
				"num_nce_per_thread":  1,
				"yolo_config": utils.AttributeMap{
					"iou_threshold":         0.5,
					"coordinate_size":       4,
					"blob_path":             absBlobPath,
					"labels": []string{
						"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
						"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
						"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
						"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
						"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
						"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
						"pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
						"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
						"teddy bear", "hair drier", "toothbrush",
					},
					"confidence_threshold": 0.5,
				},
			},
		}
		err := cam.Reconfigure(timeoutCtx, resource.Dependencies{}, cfg)
		test.That(t, err, test.ShouldBeNil)
		time.Sleep(5*time.Second)
	})

	t.Run("Capture all from camera", func(t *testing.T) {
		attempts := 0
		var capture viscapture.VisCapture
		var err error

		for attempts < 3 {
			capture, err = ydn.CaptureAllFromCamera(timeoutCtx, "", viscapture.CaptureOptions{ReturnImage: true, ReturnDetections: true, ReturnClassifications: false, ReturnObject: false}, nil)
			if err != nil {
				errStr := err.Error()
				if strings.Contains(errStr, "Camera data requested before camera worker was ready") || strings.Contains(errStr, "Could not find matching YDN config for YDN service id") {
					attempts++
					time.Sleep(1)
					continue
				} else {
					t.Errorf("Unexpected error occurred getting detections from camera: %v", errStr)
					break
				}
			}
			break
		}

		dets := capture.Detections
		test.That(t, dets, test.ShouldNotBeNil)
		for _, det := range dets {
			test.That(t, det.BoundingBox().Max.X, test.ShouldBeLessThanOrEqualTo, defaultYdnWidth)
			test.That(t, det.BoundingBox().Max.Y, test.ShouldBeLessThanOrEqualTo, defaultYdnHeight)
			test.That(t, det.Label(), test.ShouldNotBeNil)
			test.That(t, det.Label(), test.ShouldNotBeEmpty)
			test.That(t, det.Score(), test.ShouldBeLessThanOrEqualTo, 1)
		}

		img := capture.Image
		test.That(t, img, test.ShouldNotBeNil)
		bounds := img.Bounds()
		test.That(t, bounds.Dx(), test.ShouldEqual, defaultYdnWidth)
		test.That(t, bounds.Dy(), test.ShouldEqual, defaultYdnHeight)
	})

	t.Run("Shut down the camera", func(t *testing.T) {
		test.That(t, cam.Close(timeoutCtx), test.ShouldBeNil)
	})

	t.Run("Shut down the service", func(t *testing.T) {
		test.That(t, ydn.Close(timeoutCtx), test.ShouldBeNil)
	})

	t.Run("Shut down the robot", func(t *testing.T) {
		test.That(t, myRobot.Close(timeoutCtx), test.ShouldBeNil)
	})
}
