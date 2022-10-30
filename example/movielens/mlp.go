package movielens

import (
	"github.com/auxten/edgeRec/nn/base"
	nn "github.com/auxten/edgeRec/nn/neural_network"
	rcmd "github.com/auxten/edgeRec/recommend"
	"gonum.org/v1/gonum/mat"
)

type predWrap struct {
	pred base.Predicter
}

func (p *predWrap) Predict(X mat.Matrix, Y mat.Mutable) *mat.Dense {
	return p.pred.Predict(X, Y)
}

type fitWrap struct {
	model *nn.MLPClassifier
}

func (fit *fitWrap) Fit(trainSample *rcmd.TrainSample) (rcmd.PredictAbstract, error) {
	sampleLen := len(trainSample.Data)
	sampleDense := mat.NewDense(sampleLen, len(trainSample.Data[0].Input), nil)
	for i, sample := range trainSample.Data {
		sampleDense.SetRow(i, sample.Input)
	}
	yClass := mat.NewDense(sampleLen, 1, nil)
	for i, sample := range trainSample.Data {
		yClass.Set(i, 0, sample.Response[0])
	}

	pred := fit.model.Fit(sampleDense, yClass)

	return &predWrap{
		pred: pred.(base.Predicter),
	}, nil
}
