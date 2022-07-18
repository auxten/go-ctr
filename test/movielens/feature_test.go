package main

import (
	"fmt"
	"testing"

	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
)

func TestFeatureEngineer(t *testing.T) {

	var (
		recSys = &RecSysImpl{}
		model  rcmd.Predictor
		err    error
	)
	Convey("feature engineering", t, func() {
		log.SetLevel(log.DebugLevel)
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
		for _, test := range testData {
			score, err := rcmd.Rank(model, test.userId, []int{test.itemId})
			So(err, ShouldBeNil)

			fmt.Printf("userId:%d, itemId:%d, expected:%f, pred:%f\n",
				test.userId, test.itemId, test.expected, score[0].Score)
			//So(pred.At(0, 0), ShouldAlmostEqual, test.expected)
		}
	})
}
