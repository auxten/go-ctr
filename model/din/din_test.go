package din

import (
	"math"
	"math/rand"
	"testing"

	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestDin(t *testing.T) {
	Convey("Din simple Grad", t, func() {
		log.SetLevel(log.DebugLevel)
		rand.Seed(42)
		var (
			batchSize     = 100
			uProfileDim   = 9
			uBehaviorSize = 3
			uBehaviorDim  = 7
			iFeatureDim   = 7
			cFeatureDim   = 9

			numExamples = 10000
			epochs      = 200
			sampleInfo  = &rcmd.SampleInfo{
				UserProfileRange:  [2]int{0, uProfileDim},
				UserBehaviorRange: [2]int{uProfileDim, uProfileDim + uBehaviorSize*uBehaviorDim},
				ItemFeatureRange:  [2]int{uProfileDim + uBehaviorSize*uBehaviorDim, uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim},
				CtxFeatureRange:   [2]int{uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim, uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim + cFeatureDim},
			}
			inputWidth = uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim + cFeatureDim
		)
		inputSlice := make([]float64, numExamples*inputWidth)
		for i := 0; i < numExamples; i++ {
			for j := 0; j < sampleInfo.UserProfileRange[1]; j++ {
				inputSlice[i*inputWidth+j] = rand.Float64()
			}
			for j := sampleInfo.CtxFeatureRange[0]; j < sampleInfo.CtxFeatureRange[1]; j++ {
				inputSlice[i*inputWidth+j] = rand.Float64()
			}
		}
		inputs := tensor.New(tensor.WithShape(numExamples, inputWidth), tensor.WithBacking(inputSlice))
		labelSlice := make([]float64, numExamples)
		for i := 0; i < numExamples; i++ {
			//distance of uProfile and cFeature slice
			var dist float64
			for j := sampleInfo.UserProfileRange[0]; j < sampleInfo.UserProfileRange[1]; j++ {
				dist += math.Abs(inputSlice[i*inputWidth+j] - inputSlice[i*inputWidth+j+sampleInfo.CtxFeatureRange[0]])
			}
			labelSlice[i] = math.Round(dist / float64(uProfileDim) * 1.5)
		}
		labels := tensor.New(tensor.WithShape(numExamples, 1), tensor.WithBacking(labelSlice))
		log.Debugf("labels: %+v", labels.Data())
		g := G.NewGraph()
		//model := NewSimpleMLP(g, uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
		model := NewDinNet(g, uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
		err := Train(uBehaviorSize, uBehaviorDim, uProfileDim, iFeatureDim, cFeatureDim,
			numExamples, batchSize, epochs,
			sampleInfo,
			inputs, labels,
			g,
			model,
		)
		So(err, ShouldBeNil)

	})
}
