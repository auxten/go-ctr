package main

import (
	"fmt"
	"testing"

	"github.com/auxten/edgeRec/utils"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
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
			{1, 151, 1.},
			{1, 157, 1.},
			{3, 31, 0.},
			{3, 527, 0.},
			{89, 3088, 0.},
		}
		for _, test := range testData {
			userFeature := recSys.GetUserFeature(test.userId)
			itemFeature := recSys.GetItemFeature(test.itemId)
			input := utils.ConcatSlice(userFeature, itemFeature)
			pred := recSys.Neural.Predict(input)
			fmt.Printf("userId:%d, itemId:%d, expected:%f, pred:%f\n", test.userId, test.itemId, test.expected, pred[0])
			//So(pred[0], ShouldAlmostEqual, test.expected)
		}
	})
}
