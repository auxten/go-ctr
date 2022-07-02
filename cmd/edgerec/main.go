package main

import (
	"flag"
)

var (
	// Version is the version of the program
	Version = "0.0.0"
	Config  string
)

func init() {
	flag.StringVar(&Config, "config", "", "config file")
}

func main() {

}

// parseFlag parses command line options
func parseFlag() {
	flag.Parse()
}
