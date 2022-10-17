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

	batchSize, epochs int
	sampleInfo        *rcmd.SampleInfo

	model *din.DinNet
}

func (d *dinImpl) Fit(X, Y mat.Matrix) base.Fiter {
	d.model = din.NewDinNet(d.uProfileDim, d.uBehaviorSize, d.uBehaviorDim, d.iFeatureDim, d.cFeatureDim)
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
		d.model,
	)
	if err != nil {
		log.Errorf("train din model failed: %v", err)
		return nil
	}
	return d
}
