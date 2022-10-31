package movielens

import (
	"github.com/auxten/edgeRec/nn/base"
	nn "github.com/auxten/edgeRec/nn/neural_network"
	rcmd "github.com/auxten/edgeRec/recommend"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
)

type predWrap struct {
	pred base.Predicter
}

func (p *predWrap) Predict(X tensor.Tensor) tensor.Tensor {
	numPred := X.Shape()[0]
	xWidth := X.Shape()[1]
	//convert float32 tensor to float64 mat.Dense
	xDense := mat.NewDense(numPred, X.Shape()[1], nil)
	for i := 0; i < numPred; i++ {
		for j := 0; j < xWidth; j++ {
			val, err := X.At(i, j)
			if err != nil {
				return nil
			}
			xDense.Set(i, j, float64(val.(float32)))
		}
	}
	yMutable := mat.NewDense(numPred, 1, nil)
	p.pred.Predict(xDense, yMutable)

	//convert float64 mat.Dense to float32 tensor
	y := make([]float32, numPred)
	for i := 0; i < numPred; i++ {
		val := yMutable.At(i, 0)
		y[i] = float32(val)
	}
	return tensor.NewDense(tensor.Float32, tensor.Shape{numPred, 1}, tensor.WithBacking(y))
}

type fitWrap struct {
	model *nn.MLPClassifier
}

func (fit *fitWrap) Fit(trainSample *rcmd.TrainSample) (rcmd.PredictAbstract, error) {
	sampleLen := trainSample.Rows
	x64 := make([]float64, sampleLen*trainSample.XCols)
	for i := 0; i < sampleLen; i++ {
		for j := 0; j < trainSample.XCols; j++ {
			x64[i*trainSample.XCols+j] = float64(trainSample.X[i*trainSample.XCols+j])
		}
	}
	sampleDense := mat.NewDense(sampleLen, trainSample.XCols, x64)

	yClass := mat.NewDense(sampleLen, 1, nil)
	for i, sample := range trainSample.Y {
		yClass.Set(i, 0, float64(sample))
	}

	pred := fit.model.Fit(sampleDense, yClass)

	return &predWrap{
		pred: pred.(base.Predicter),
	}, nil
}
