package movielens

import (
	"github.com/auxten/edgeRec/model/din"
	"github.com/auxten/edgeRec/nn/base"
	rcmd "github.com/auxten/edgeRec/recommend"
	log "github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"
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

	learner *din.DinNet
	pred    *din.DinNet
}

func (d *dinImpl) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
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

func (d *dinImpl) Fit(X, Y mat.Matrix) base.Fiter {
	d.learner = din.NewDinNet(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim)
	var (
		inputs, labels tensor.Tensor
		numExamples, _ = X.Dims()
		numLabels, _   = Y.Dims()
	)
	if numExamples != numLabels {
		log.Errorf("X and Y should have same number of rows")
		return nil
	}

	inputs = tensor.New(tensor.WithShape(X.Dims()), tensor.WithBacking(X.(*mat.Dense).RawMatrix().Data))
	labels = tensor.New(tensor.WithShape(Y.Dims()), tensor.WithBacking(Y.(*mat.Dense).RawMatrix().Data))
	err := din.Train(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim,
		numExamples, d.batchSize, d.epochs,
		d.sampleInfo,
		inputs, labels,
		d.learner,
	)
	if err != nil {
		log.Errorf("train din model failed: %v", err)
		return nil
	}
	dinJson, err := d.learner.Marshal()
	if err != nil {
		log.Errorf("marshal din model failed: %v", err)
		return nil
	}
	predictor, err := din.NewDinNetFromJson(dinJson)
	if err != nil {
		log.Errorf("new din model from json failed: %v", err)
		return nil
	}
	err = din.InitForwardOnlyVm(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim, d.predBatchSize, predictor)
	if err != nil {
		log.Errorf("init forward only vm failed: %v", err)
		return nil
	}
	d.pred = predictor

	return d
}
