package tests

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.viam.com/rdk/config"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/robot"
	robotimpl "go.viam.com/rdk/robot/impl"
)

var (
	modulePath = flag.String("module", "", "the path to the OAK module to test. If blank, will test the module from the registry.")
	blobPath   = flag.String("blob", "", "the path to the model blob. This path can be relative or absolute.")
	absBlobPath = ""
	absModulePath = ""
)

func TestMain(m *testing.M) {
	fmt.Println("OAK MODULE INTEGRATION TESTS")
	flag.Parse()

	moduleString := strings.TrimSpace(*modulePath)
	blobString := strings.TrimSpace(*blobPath)

	if moduleString == "" {
		fmt.Println("The path to the module is a required argument e.g. $ ./oak-integration-tests -module /path/to/module")
		os.Exit(1)
	}

	// Resolve absolute path for modulePath
	absPath, err := filepath.Abs(moduleString)
	if err != nil {
		fmt.Printf("  error resolving absolute path for module: %v\n", err.Error())
		os.Exit(1)
	}
	absModulePath = absPath

	// Check if module exists
	fmt.Printf("Checking if file exists at %q\n", absModulePath)
	_, err = os.Stat(absModulePath)
	if err != nil {
		fmt.Printf("  error: %v\n", err.Error())
		os.Exit(1)
	}
	fmt.Print("File exists.")

	// Resolve absolute path for blobPath if provided
	if blobString != "" {
		absBlobPath, err = filepath.Abs(blobString)
		if err != nil {
			fmt.Printf("  error resolving absolute path for blob: %v\n", err.Error())
			os.Exit(1)
		}

		// Check if blob exists
		fmt.Printf("Checking if model blob exists at %q\n", absBlobPath)
		_, err = os.Stat(absBlobPath)
		if err != nil {
			fmt.Printf("  error: %v\n", err.Error())
			os.Exit(1)
		}
		fmt.Print("Blob exists.")
	} else {
		fmt.Print("No blob path provided.")
	}

	exitVal := m.Run()
	if exitVal == 0 {
		fmt.Println("All tests succeeded!")
	}
	os.Exit(exitVal)
}

func setUpViamServer(ctx context.Context, configString string, loggerName string, _ *testing.T) (robot.Robot, error) {
	logger := logging.NewLogger(loggerName)

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
