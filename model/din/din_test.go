package din

import (
	"testing"

	rcmd "github.com/auxten/edgeRec/recommend"
	. "github.com/smartystreets/goconvey/convey"
	"gorgonia.org/tensor"
)

func TestDin(t *testing.T) {
	Convey("Din simple Grad", t, func() {
		var (
			batchSize     = 10
			uProfileDim   = 2
			uBehaviorSize = 3
			uBehaviorDim  = 7
			iFeatureDim   = 7
			cFeatureDim   = 9

			numExamples = 20
			epochs      = 100
			sampleInfo  = &rcmd.SampleInfo{
				UserProfileRange:  [2]int{0, uProfileDim},
				UserBehaviorRange: [2]int{uProfileDim, uProfileDim + uBehaviorSize*uBehaviorDim},
				ItemFeatureRange:  [2]int{uProfileDim + uBehaviorSize*uBehaviorDim, uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim},
				CtxFeatureRange:   [2]int{uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim, uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim + cFeatureDim},
			}
			inputWidth = uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim + cFeatureDim
		)
		inputSlice := make([]float64, numExamples*inputWidth)
		for i := range inputSlice {
			inputSlice[i] = float64(i)/float64(numExamples*inputWidth) - 0.5
		}
		inputs := tensor.New(tensor.WithShape(numExamples, inputWidth), tensor.WithBacking(inputSlice))
		labelSlice := make([]float64, numExamples)
		for i := 0; i < numExamples; i++ {
			labelSlice[i] = float64(i % 2)
		}
		labels := tensor.New(tensor.WithShape(numExamples, 1), tensor.WithBacking(labelSlice))

		err := Train(uBehaviorSize, uBehaviorDim, uProfileDim, iFeatureDim, cFeatureDim,
			numExamples, batchSize, epochs,
			sampleInfo,
			inputs, labels,
		)
		So(err, ShouldBeNil)

	})
}
