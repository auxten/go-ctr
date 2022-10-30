package movielens

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/auxten/edgeRec/nn/metrics"
	rcmd "github.com/auxten/edgeRec/recommend"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

type dinPredictor struct {
	rcmd.PreRanker
	rcmd.Predictor
	rcmd.UserBehavior
}

func TestDinOnMovielens(t *testing.T) {
	rand.Seed(42)

	var (
		movielens = &MovielensRec{
			DataPath:  "movielens.db",
			SampleCnt: 79948,
			//SampleCnt: 10000,
		}
		model rcmd.Predictor
		err   error
	)

	Convey("Train din model", t, func() {
		dinModel := &dinImpl{
			predBatchSize: 100,
			batchSize:     200,
			epochs:        100,
			earlyStop:     20,
		}
		trainCtx := context.Background()
		model, err = rcmd.Train(trainCtx, movielens, dinModel)
		So(err, ShouldBeNil)
		So(model, ShouldNotBeNil)
	})

	Convey("Predict din model", t, func() {
		testCount := 20600
		rows, err := db.Query(
			"SELECT userId, movieId, rating, timestamp FROM ratings_test ORDER BY timestamp, userId ASC LIMIT ?", testCount)
		So(err, ShouldBeNil)
		var (
			userId     int
			itemId     int
			rating     float64
			timestamp  int64
			yTrue      = mat.NewDense(testCount, 1, nil)
			sampleKeys = make([]rcmd.Sample, 0, testCount)
		)
		for i := 0; rows.Next(); i++ {
			err = rows.Scan(&userId, &itemId, &rating, &timestamp)
			if err != nil {
				t.Errorf("scan error: %v", err)
			}
			yTrue.Set(i, 0, BinarizeLabel(rating))
			sampleKeys = append(sampleKeys, rcmd.Sample{userId, itemId, 0, timestamp})
		}
		batchPredictCtx := context.Background()
		dinPred := &dinPredictor{
			PreRanker:    movielens,
			Predictor:    model,
			UserBehavior: movielens,
		}
		yPred, err := rcmd.BatchPredict(batchPredictCtx, dinPred, sampleKeys)
		So(err, ShouldBeNil)
		rocAuc := metrics.ROCAUCScore(yTrue, yPred, "", nil)
		rowCount, _ := yTrue.Dims()
		fmt.Printf("rocAuc on test set %d: %f\n", rowCount, rocAuc)
	})
}
