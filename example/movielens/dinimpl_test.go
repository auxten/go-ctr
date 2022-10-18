package movielens

import (
	"context"
	"fmt"
	"testing"

	"github.com/auxten/edgeRec/nn/metrics"
	rcmd "github.com/auxten/edgeRec/recommend"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestDinOnMovielens(t *testing.T) {
	var (
		movielens = &MovielensRec{
			DataPath: "movielens.db",
			//SampleCnt: 79948,
			SampleCnt: 1000,
		}
		model rcmd.Predictor
		err   error
	)

	Convey("Train din model", t, func() {
		dinModel := &dinImpl{
			predBatchSize: 100,
			batchSize:     100,
			epochs:        1,
		}
		trainCtx := context.Background()
		model, err = rcmd.Train(trainCtx, movielens, dinModel)
		So(err, ShouldBeNil)
		So(model, ShouldNotBeNil)
	})

	Convey("Predict din model", t, func() {
		testCount := 20600
		rows, err := db.Query(
			"SELECT userId, movieId, rating FROM ratings_test ORDER BY timestamp, userId ASC LIMIT ?", testCount)
		So(err, ShouldBeNil)
		var (
			userId       int
			itemId       int
			rating       float64
			yTrue        = mat.NewDense(testCount, 1, nil)
			userAndItems [][2]int
		)
		for i := 0; rows.Next(); i++ {
			err = rows.Scan(&userId, &itemId, &rating)
			if err != nil {
				t.Errorf("scan error: %v", err)
			}
			yTrue.Set(i, 0, BinarizeLabel(rating))
			userAndItems = append(userAndItems, [2]int{userId, itemId})
		}
		batchPredictCtx := context.Background()
		yPred, err := rcmd.BatchPredict(batchPredictCtx, model, userAndItems)
		So(err, ShouldBeNil)
		rocAuc := metrics.ROCAUCScore(yTrue, yPred, "", nil)
		rowCount, _ := yTrue.Dims()
		fmt.Printf("rocAuc on test set %d: %f\n", rowCount, rocAuc)

	})
}
