package movielens

import (
	"fmt"

	"github.com/auxten/edgeRec/model"
	"github.com/auxten/edgeRec/model/youtube"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	"gorgonia.org/tensor"
)

type YoutubeDnnImpl struct {
	uProfileDim   int
	uBehaviorSize int
	uBehaviorDim  int
	iFeatureDim   int
	cFeatureDim   int

	predBatchSize     int
	batchSize, epochs int
	sampleInfo        *rcmd.SampleInfo

	// stop training on earlyStop count of no cost improvement
	// 0 means no early stop
	earlyStop int

	learner *youtube.YoutubeDnn
	pred    *youtube.YoutubeDnn
}

func (d *YoutubeDnnImpl) Predict(X tensor.Tensor) tensor.Tensor {
	numPred := X.Shape()[0]
	y, err := model.Predict(d.pred, numPred, d.predBatchSize, d.sampleInfo, X)
	if err != nil {
		log.Errorf("predict din model failed: %v", err)
		return nil
	}
	yDense := tensor.NewDense(model.DT, tensor.Shape{numPred, 1}, tensor.WithBacking(y))

	return yDense
}

func (d *YoutubeDnnImpl) Fit(trainSample *rcmd.TrainSample) (pred rcmd.PredictAbstract, err error) {
	d.uProfileDim = trainSample.Info.UserProfileRange[1] - trainSample.Info.UserProfileRange[0]
	d.uBehaviorSize = rcmd.UserBehaviorLen
	d.uBehaviorDim = rcmd.ItemEmbDim
	d.iFeatureDim = rcmd.ItemEmbDim
	d.cFeatureDim = trainSample.Info.CtxFeatureRange[1] - trainSample.Info.CtxFeatureRange[0]
	d.sampleInfo = &trainSample.Info

	if trainSample.Rows != len(trainSample.Y) {
		err = fmt.Errorf("number of examples and labels do not match")
		return
	}

	d.learner = youtube.NewYoutubeDnn(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim)

	inputs := tensor.New(tensor.WithShape(trainSample.Rows, trainSample.XCols), tensor.WithBacking(trainSample.X))
	labels := tensor.New(tensor.WithShape(trainSample.Rows, 1), tensor.WithBacking(trainSample.Y))
	err = model.Train(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim,
		trainSample.Rows, d.batchSize, d.epochs, d.earlyStop,
		d.sampleInfo,
		inputs, labels,
		d.learner,
	)
	if err != nil {
		log.Errorf("train din model failed: %v", err)
		return
	}
	dinJson, err := d.learner.Marshal()
	if err != nil {
		log.Errorf("marshal din model failed: %v", err)
		return
	}
	//log.Debugf("din model json: %s", dinJson)
	dinPred, err := youtube.NewYoutubeDnnFromJson(dinJson)
	if err != nil {
		log.Errorf("new din model from json failed: %v", err)
		return
	}
	err = model.InitForwardOnlyVm(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim,
		d.predBatchSize, dinPred)
	if err != nil {
		log.Errorf("init forward only vm failed: %v", err)
		return
	}
	d.pred = dinPred

	return d, nil
}
