package movielens

import (
	"context"
	"fmt"
	"testing"

	"github.com/auxten/edgeRec/nn/metrics"
	nn "github.com/auxten/edgeRec/nn/neural_network"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestFeatureEngineer(t *testing.T) {
	rcmd.DebugUserId = 429
	rcmd.DebugItemId = 588

	var (
		recSys = &MovielensRec{
			DataPath:  "movielens.db",
			SampleCnt: 79948,
		}
		model rcmd.Predictor
		err   error
	)

	fitter := nn.NewMLPClassifier(
		[]int{100},
		"relu", "adam", 1e-5,
	)
	fitter.Verbose = true
	fitter.MaxIter = 10
	fitter.LearningRate = "adaptive"
	fitter.LearningRateInit = .0025

	trainCtx := context.Background()
	log.SetLevel(log.DebugLevel)
	Convey("feature engineering", t, func() {
		model, err = rcmd.Train(trainCtx, recSys, &fitWrap{model: fitter})
		So(err, ShouldBeNil)
	})

	Convey("prediction", t, func() {
		testData := []struct {
			userId   int
			itemId   int
			expected float64
		}{
			{8, 527, 1.},
			{8, 432, 0.},
			{106, 318, 1.},
			{106, 31696, 0.},
			{111, 588, 1.},
			{111, 51086, 0.},
		}

		var (
			yTrue = mat.NewDense(len(testData), 1, nil)
			yPred = mat.NewDense(len(testData), 1, nil)
		)
		rankCtx := context.Background()
		for i, test := range testData {
			score, err := rcmd.Rank(rankCtx, model, test.userId, []int{test.itemId})
			So(err, ShouldBeNil)

			fmt.Printf("userId:%d, itemId:%d, expected:%f, pred:%f\n",
				test.userId, test.itemId, test.expected, score[0].Score)
			//So(pred.At(0, 0), ShouldAlmostEqual, test.expected)
			yTrue.Set(i, 0, test.expected)
			yPred.Set(i, 0, score[0].Score)
		}

		rocAuc := metrics.ROCAUCScore(yTrue, yPred, "", nil)
		fmt.Printf("rocAuc:%f\n", rocAuc)
	})

	Convey("test set ROC AUC", t, func() {
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
		yPred, err := rcmd.BatchPredict(batchPredictCtx, model, sampleKeys)
		So(err, ShouldBeNil)
		rocAuc := metrics.ROCAUCScore(yTrue, yPred, "", nil)
		rowCount, _ := yTrue.Dims()
		fmt.Printf("rocAuc on test set %d: %f\n", rowCount, rocAuc)
	})
}
