package movielens

import (
	"fmt"

	"github.com/auxten/edgeRec/model/din"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	"gorgonia.org/tensor"
)

type dinImpl struct {
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

	learner *din.DinNet
	pred    *din.DinNet
}

func (d *dinImpl) Predict(X tensor.Tensor) tensor.Tensor {
	numPred := X.Shape()[0]
	y, err := din.Predict(d.pred, numPred, d.predBatchSize, d.sampleInfo, X)
	if err != nil {
		log.Errorf("predict din model failed: %v", err)
		return nil
	}
	yDense := tensor.NewDense(din.DT, tensor.Shape{numPred, 1}, tensor.WithBacking(y))

	return yDense
}

func (d *dinImpl) Fit(trainSample *rcmd.TrainSample) (pred rcmd.PredictAbstract, err error) {
	d.uProfileDim = trainSample.Info.UserProfileRange[1] - trainSample.Info.UserProfileRange[0]
	d.uBehaviorSize = rcmd.UserBehaviorLen
	d.uBehaviorDim = rcmd.ItemEmbDim
	d.iFeatureDim = rcmd.ItemEmbDim
	d.cFeatureDim = trainSample.Info.CtxFeatureRange[1] - trainSample.Info.CtxFeatureRange[0]
	d.sampleInfo = &trainSample.Info

	if trainSample.Rows != len(trainSample.Y) {
		err = fmt.Errorf("number of examples %d and labels %d do not match",
			trainSample.Rows, len(trainSample.Y))
		return
	}

	inputs := tensor.New(tensor.WithShape(trainSample.Rows, trainSample.XCols), tensor.WithBacking(trainSample.X))
	labels := tensor.New(tensor.WithShape(trainSample.Rows, 1), tensor.WithBacking(trainSample.Y))

	d.learner = din.NewDinNet(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim)

	err = din.Train(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim,
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
	dinPred, err := din.NewDinNetFromJson(dinJson)
	if err != nil {
		log.Errorf("new din model from json failed: %v", err)
		return
	}
	err = din.InitForwardOnlyVm(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim,
		d.predBatchSize, dinPred)
	if err != nil {
		log.Errorf("init forward only vm failed: %v", err)
		return
	}
	d.pred = dinPred

	return d, nil
}
