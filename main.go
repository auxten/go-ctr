package main

import (
	"github.com/auxten/edgeRec/example/movielens"
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
	model, err = rcmd.Train(recSys)
	if err != nil {
		log.Fatal(err)
	}
	rcmd.StartHttpApi(model, "/api/v1/recommend", ":8080")
}
