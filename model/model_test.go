package model_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/auxten/edgeRec/model"
	"github.com/auxten/edgeRec/model/din"
	"github.com/auxten/edgeRec/model/youtube"
	rcmd "github.com/auxten/edgeRec/recommend"
	"github.com/auxten/edgeRec/utils"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
	"gorgonia.org/tensor"
)

func TestMultiModel(t *testing.T) {
	log.SetLevel(log.DebugLevel)
	rand.Seed(42)
	var (
		batchSize     = 200
		uProfileDim   = 5
		uBehaviorSize = 3
		uBehaviorDim  = 7
		iFeatureDim   = 7
		cFeatureDim   = 5

		numExamples = 100000
		epochs      = 20

		// to test sample count not fully match batch size
		testSamples   = 118
		testBatchSize = 20

		sampleInfo = &rcmd.SampleInfo{
			UserProfileRange:  [2]int{0, uProfileDim},
			UserBehaviorRange: [2]int{uProfileDim, uProfileDim + uBehaviorSize*uBehaviorDim},
			ItemFeatureRange:  [2]int{uProfileDim + uBehaviorSize*uBehaviorDim, uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim},
			CtxFeatureRange:   [2]int{uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim, uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim + cFeatureDim},
		}
		inputWidth = uProfileDim + uBehaviorSize*uBehaviorDim + iFeatureDim + cFeatureDim
	)
	inputSlice := make([]float32, numExamples*inputWidth)
	for i := 0; i < numExamples; i++ {
		for j := 0; j < sampleInfo.UserProfileRange[1]; j++ {
			inputSlice[i*inputWidth+j] = rand.Float32()
		}
		for j := sampleInfo.CtxFeatureRange[0]; j < sampleInfo.CtxFeatureRange[1]; j++ {
			inputSlice[i*inputWidth+j] = rand.Float32()
		}
		for j := sampleInfo.UserBehaviorRange[0] + uBehaviorDim; j < sampleInfo.UserBehaviorRange[0]+2*uBehaviorDim; j++ {
			inputSlice[i*inputWidth+j] = rand.Float32()
		}
		for j := sampleInfo.ItemFeatureRange[0]; j < sampleInfo.ItemFeatureRange[1]; j++ {
			inputSlice[i*inputWidth+j] = rand.Float32()
		}
	}
	//for i := 0; i < numExamples*inputWidth; i++ {
	//	inputSlice[i] = rand.Float32()
	//}
	inputs := tensor.New(tensor.WithShape(numExamples, inputWidth), tensor.WithBacking(inputSlice))
	labelSlice := make([]float32, numExamples)
	for i := 0; i < numExamples; i++ {
		//distance of uProfile and cFeature slice
		var dist1, dist2 float32
		for j := 0; j < uProfileDim; j++ {
			dist1 += float32(math.Abs(float64(inputSlice[i*inputWidth+sampleInfo.UserProfileRange[0]+j] - inputSlice[i*inputWidth+sampleInfo.CtxFeatureRange[0]+j])))
		}
		labelSlice[i] = dist1 / float32(uProfileDim)

		//distance of 2nd uBehavior and iFeature
		for j := 0; j < uBehaviorDim; j++ {
			dist2 += float32(math.Abs(float64(inputSlice[i*inputWidth+sampleInfo.UserBehaviorRange[0]+uBehaviorDim+j] - inputSlice[i*inputWidth+sampleInfo.ItemFeatureRange[0]+j])))
		}
		labelSlice[i] = float32(math.Round(float64((labelSlice[i] + (dist2 / float32(uBehaviorDim))) * 0.6)))
	}

	labels := tensor.New(tensor.WithShape(numExamples, 1), tensor.WithBacking(labelSlice))
	//log.Debugf("labels: %+v", labels.Data())

	dinModel := din.NewDinNet(uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
	Convey("Din model", t, func() {
		err := model.Train(uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim,
			numExamples, batchSize, epochs, 0,
			sampleInfo,
			inputs, labels,
			dinModel,
		)
		So(err, ShouldBeNil)
	})

	var dinPredict *din.DinNet
	Convey("Din model marshal and new from json", t, func() {
		dinJson, err := dinModel.Marshal()
		So(err, ShouldBeNil)
		//log.Debugf("dinJson: %s", dinJson)
		dinPredict, err = din.NewDinNetFromJson(dinJson)
		So(err, ShouldBeNil)
	})

	Convey("Din predict", t, func() {
		err := model.InitForwardOnlyVm(uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim, testBatchSize, dinPredict)
		So(err, ShouldBeNil)
		predictions, err := model.Predict(dinPredict, testSamples, testBatchSize, sampleInfo, inputs)
		So(err, ShouldBeNil)
		So(predictions, ShouldNotBeNil)
		So(predictions, ShouldHaveLength, testSamples)
		log.Debugf("predictions: %+v", predictions)
		auc := utils.RocAuc32(predictions, labels.Data().([]float32)[:testSamples])
		log.Debugf("auc: %f", auc)
		So(auc, ShouldBeGreaterThan, 0.5)
	})

	youtubeDnnModel := youtube.NewYoutubeDnn(uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim)
	Convey("Youtube DNN", t, func() {
		err := model.Train(uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim,
			numExamples, batchSize, epochs, 10,
			sampleInfo,
			inputs, labels,
			youtubeDnnModel,
		)
		So(err, ShouldBeNil)
	})

	var yDnnPredict *youtube.YoutubeDnn
	Convey("Youtube DNN marshal and new from json", t, func() {
		mlpJson, err := youtubeDnnModel.Marshal()
		So(err, ShouldBeNil)
		//log.Debugf("mlpJson: %s", mlpJson)
		yDnnPredict, err = youtube.NewYoutubeDnnFromJson(mlpJson)
		So(err, ShouldBeNil)
	})

	Convey("Youtube DNN predict", t, func() {
		err := model.InitForwardOnlyVm(uProfileDim, uBehaviorSize, uBehaviorDim, iFeatureDim, cFeatureDim, testBatchSize, yDnnPredict)
		So(err, ShouldBeNil)
		predictions, err := model.Predict(yDnnPredict, testSamples, testBatchSize, sampleInfo, inputs)
		So(err, ShouldBeNil)
		So(predictions, ShouldNotBeNil)
		So(predictions, ShouldHaveLength, testSamples)
		log.Debugf("predictions: %+v", predictions)
		auc := utils.RocAuc32(predictions, labels.Data().([]float32)[:testSamples])
		log.Debugf("auc: %f", auc)
		So(auc, ShouldBeGreaterThan, 0.5)
	})
}
