package main

import (
	"fmt"
	"testing"

	"github.com/auxten/edgeRec/nn/metrics"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestFeatureEngineer(t *testing.T) {
	var (
		recSys = &RecSysImpl{}
		model  rcmd.Predictor
		err    error
	)

	log.SetLevel(log.DebugLevel)
	Convey("feature engineering", t, func() {
		model, err = rcmd.Train(recSys)
		So(err, ShouldBeNil)
	})

	Convey("prediction", t, func() {
		testData := []struct {
			userId   int
			itemId   int
			expected float64
		}{
			{429, 588, 1.},
			{429, 22, 1.},
			{107, 1, 1.},
			{107, 2, 1.},
			{191, 39, 0.},
			{11, 1391, 0.},
		}

		var (
			yTrue = mat.NewDense(len(testData), 1, nil)
			yPred = mat.NewDense(len(testData), 1, nil)
		)
		for i, test := range testData {
			score, err := rcmd.Rank(model, test.userId, []int{test.itemId})
			So(err, ShouldBeNil)

			fmt.Printf("userId:%d, itemId:%d, expected:%f, pred:%f\n",
				test.userId, test.itemId, test.expected, score[0].Score)
			//So(pred.At(0, 0), ShouldAlmostEqual, test.expected)
			yTrue.Set(i, 0, test.expected)
			yPred.Set(i, 0, score[0].Score)
		}

		rocAuc := metrics.ROCAUCScore(yTrue, yPred, "", nil)
		fmt.Printf("rocAuc:%f\n", rocAuc)
		//So(rocAuc, ShouldBeGreaterThan, 0.7)
	})

	Convey("test set ROC AUC", t, func() {
		testCount := 20836
		rows, err := db.Query(
			"SELECT userId, movieId, rating FROM ratings_test LIMIT ?", testCount)
		So(err, ShouldBeNil)
		var (
			userId int
			itemId int
			rating float64
			yTrue  = mat.NewDense(testCount, 1, nil)
			yPred  = mat.NewDense(testCount, 1, nil)
		)
		for i := 0; rows.Next(); i++ {
			err = rows.Scan(&userId, &itemId, &rating)
			if err != nil {
				t.Errorf("scan error: %v", err)
			}
			yTrue.Set(i, 0, BinarizeLabel(rating))
			score, err := rcmd.Rank(model, userId, []int{itemId})
			if err != nil {
				t.Errorf("rank error: %v", err)
			}
			yPred.Set(i, 0, score[0].Score)
		}
		rocAuc := metrics.ROCAUCScore(yTrue, yPred, "", nil)
		rowCount, _ := yTrue.Dims()
		fmt.Printf("rocAuc on test set %d: %f\n", rowCount, rocAuc)
	})
}
