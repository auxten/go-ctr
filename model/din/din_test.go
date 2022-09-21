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
			inputs     = tensor.New(tensor.WithShape(numExamples, inputWidth), tensor.WithBacking(tensor.Range(tensor.Float64, 0, inputWidth*numExamples)))
			labels     = tensor.New(tensor.WithShape(numExamples, 1), tensor.WithBacking(tensor.Range(tensor.Float64, 0, numExamples)))
		)

		err := Train(uBehaviorSize, uBehaviorDim, uProfileDim, iFeatureDim, cFeatureDim,
			numExamples, batchSize, epochs,
			sampleInfo,
			inputs, labels,
		)
		So(err, ShouldBeNil)

	})
}
