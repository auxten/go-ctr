package main

import (
	"github.com/auxten/edgeRec/example/movielens"
	nn "github.com/auxten/edgeRec/nn/neural_network"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
)

func main() {
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

	model, err = rcmd.Train(recSys, fiter)
	if err != nil {
		log.Fatal(err)
	}
	rcmd.StartHttpApi(model, "/api/v1/recommend", ":8080")
}
