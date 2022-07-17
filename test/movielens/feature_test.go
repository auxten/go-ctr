package main

import (
	"fmt"
	"testing"

	"github.com/auxten/edgeRec/utils"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestFeatureEngineer(t *testing.T) {
	recSys := &RecSysImpl{}
	Convey("feature engineering", t, func() {
		log.SetLevel(log.DebugLevel)
		err := Train(recSys)
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
			userFeature := recSys.GetUserFeature(test.userId)
			itemFeature := recSys.GetItemFeature(test.itemId)
			xSlice := utils.ConcatSlice(userFeature, itemFeature)
			x := mat.NewDense(1, len(xSlice), xSlice)
			y := mat.NewDense(1, 1, nil)

			pred := recSys.Neural.Predict(x, y)
			fmt.Printf("userId:%d, itemId:%d, expected:%f, pred:%f\n",
				test.userId, test.itemId, test.expected, pred.At(0, 0))
			//So(pred.At(0, 0), ShouldAlmostEqual, test.expected)
		}
	})
}
