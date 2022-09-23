package main

import (
	"context"
	"embed"
	"flag"
	"github.com/auxten/edgeRec/example/movielens"
	nn "github.com/auxten/edgeRec/nn/neural_network"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
)

//go:embed frontend/website/*
var f embed.FS

var verFlag = flag.Bool("v", false, "show binary version")

var Version = "unknown-version"
var Commit = "unknown-commit"

func main() {
	flag.Parse()
	if *verFlag {
		log.Println("Version: ", Version)
		log.Println("Commit: ", Commit)
		return
	}
	
	var (
		recSys = &movielens.RecSysImpl{
			DataPath:  "movielens.db",
			SampleCnt: 80000,
		}

		model rcmd.Predictor
		err   error
	)
	log.SetLevel(log.DebugLevel)

	fiter := nn.NewMLPClassifier(
		[]int{100},
		"relu", "adam", 1e-5,
	)
	fiter.Verbose = true
	fiter.MaxIter = 20
	// fiter.LearningRate = "adaptive"
	// fiter.LearningRateInit = .0025

	trainCtx := context.Background()
	model, err = rcmd.Train(trainCtx, recSys, fiter)
	if err != nil {
		log.Fatal(err)
	}
	rcmd.StartHttpApi(model, "/api/v1/recommend", ":8080", &f)
}
