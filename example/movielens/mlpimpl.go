package movielens

import (
	"fmt"

	"github.com/auxten/edgeRec/model/din"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
)

type mlpImpl struct {
	uProfileDim   int
	uBehaviorSize int
	uBehaviorDim  int
	iFeatureDim   int
	cFeatureDim   int

	predBatchSize     int
	batchSize, epochs int
	sampleInfo        *rcmd.SampleInfo

	learner *din.SimpleMLP
	pred    *din.SimpleMLP
}

func (d *mlpImpl) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	numPred, _ := X.Dims()
	inputTensor := tensor.New(tensor.WithShape(X.Dims()), tensor.WithBacking(X.(*mat.Dense).RawMatrix().Data))
	y, err := din.Predict(d.pred, numPred, d.predBatchSize, d.sampleInfo, inputTensor)
	if err != nil {
		log.Errorf("predict din model failed: %v", err)
		return nil
	}
	yDense := mat.NewDense(numPred, 1, y)
	if Y != nil {
		Y.(*mat.Dense).SetRawMatrix(yDense.RawMatrix())
	}

	return yDense
}

func (d *mlpImpl) Fit(trainSample *rcmd.TrainSample) (pred rcmd.PredictAbstract, err error) {
	d.uProfileDim = trainSample.Info.UserProfileRange[1] - trainSample.Info.UserProfileRange[0]
	d.uBehaviorSize = rcmd.UserBehaviorLen
	d.uBehaviorDim = rcmd.ItemEmbDim
	d.iFeatureDim = rcmd.ItemEmbDim
	d.cFeatureDim = trainSample.Info.CtxFeatureRange[1] - trainSample.Info.CtxFeatureRange[0]
	d.sampleInfo = &trainSample.Info

	sampleLen := len(trainSample.Data)
	X := mat.NewDense(sampleLen, len(trainSample.Data[0].Input), nil)
	for i, sample := range trainSample.Data {
		X.SetRow(i, sample.Input)
	}
	Y := mat.NewDense(sampleLen, 1, nil)
	for i, sample := range trainSample.Data {
		Y.Set(i, 0, sample.Response[0])
	}

	d.learner = din.NewSimpleMLP(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim)
	var (
		inputs, labels tensor.Tensor
		numExamples, _ = X.Dims()
		numLabels, _   = Y.Dims()
	)
	if numExamples != numLabels {
		err = fmt.Errorf("number of examples and labels do not match")
		return
	}

	inputs = tensor.New(tensor.WithShape(X.Dims()), tensor.WithBacking(X.RawMatrix().Data))
	labels = tensor.New(tensor.WithShape(Y.Dims()), tensor.WithBacking(Y.RawMatrix().Data))
	err = din.Train(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim,
		numExamples, d.batchSize, d.epochs,
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
	dinPred, err := din.NewSimpleMLPFromJson(dinJson)
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
