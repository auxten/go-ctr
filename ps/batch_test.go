package ps

import (
	"math/rand"
	"runtime"
	"testing"

	"github.com/auxten/edgeRec/nn"
)

func Benchmark_xor(b *testing.B) {
	rand.Seed(0)
	n := nn.NewNeural(&nn.Config{
		Inputs:     2,
		Layout:     []int{32, 32, 1},
		Activation: nn.ActivationSigmoid,
		Mode:       nn.ModeBinary,
		Weight:     nn.NewUniform(.25, 0),
		Bias:       true,
	})
	exs := Samples{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}
	const minSamples = 4000
	var dupExs Samples
	for len(dupExs) < minSamples {
		dupExs = append(dupExs, exs...)
	}

	for i := 0; i < b.N; i++ {
		const iterations = 20
		solver := NewAdam(0.001, 0.9, 0.999, 1e-8)
		trainer := NewBatchTrainer(solver, iterations, len(dupExs)/2, runtime.NumCPU())
		trainer.Train(n, dupExs, dupExs, iterations, true)
	}
}
