package tests

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"testing"
)

var modulePath = flag.String("module", "", "the path to the OAK module to test. If blank, will test the module from the registry.")

func TestMain(m *testing.M) {
	fmt.Println("OAK MODULE INTEGRATION TESTS")
	flag.Parse()
	moduleString := strings.TrimSpace(*modulePath)
	if moduleString == "" {
		fmt.Println("The path to the module is a required argument e.g. $ ./oak-integration-tests -module /path/to/module")
		os.Exit(1)
	}
	// check if module even exists
	fmt.Printf("Checking if file exists at %q\n", moduleString)
	_, err := os.Stat(moduleString)
	if err != nil {
		fmt.Printf("  error: %v\n", err.Error())
		os.Exit(1)
	}
	fmt.Print("File exists.")
	exitVal := m.Run()
	if exitVal == 0 {
		fmt.Println("All tests succeeded!")
	}
	os.Exit(exitVal)
}
