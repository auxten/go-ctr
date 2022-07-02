package feature

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKBinsDiscretizerTransform(t *testing.T) {
	samples := []struct {
		name      string
		quantiles []float64
		input     float64
		output    float64
	}{
		{"basic1", []float64{25, 50, 75, 100}, 0, 1},
		{"basic2", []float64{25, 50, 75, 100}, 11, 1},
		{"basic3", []float64{25, 50, 75, 100}, 25, 1},
		{"basic4", []float64{25, 50, 75, 100}, 40, 2},
		{"basic5", []float64{25, 50, 75, 100}, 50, 2},
		{"basic6", []float64{25, 50, 75, 100}, 80, 4},
		{"above_max", []float64{25, 50, 75, 100}, 101, 5},
		{"empty", nil, 10, 0},
	}
	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := KBinsDiscretizer{QuantileScaler{Quantiles: s.quantiles}}
			features := encoder.Transform((s.input))
			assert.Equal(t, s.output, features)
		})
	}
}

func TestKBinsDiscretizerTransformFit(t *testing.T) {
	samples := []struct {
		name      string
		quantiles []float64
		vals      []float64
	}{
		{"noinput", nil, nil},
		{"basic", []float64{25, 50, 75, 100}, []float64{25, 50, 75, 100}},
		{"reverse_order", []float64{25, 50, 75, 100}, []float64{100, 75, 50, 25}},
		{"negative", []float64{-100, -75, -50, -25}, []float64{-25, -50, -75, -100}},
		{"one_element", []float64{10}, []float64{10}},
		{"less_elements_than_quantiles", []float64{1, 2, 3}, []float64{1, 2, 3}},
		{"less_elements_than_quantiles_negative", []float64{-3, -2, -1}, []float64{-1, -3, -2}},
	}
	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := KBinsDiscretizer{QuantileScaler{}}
			encoder.Fit(s.vals)
			assert.Equal(t, KBinsDiscretizer{QuantileScaler{Quantiles: s.quantiles}}, encoder)
		})
	}

	t.Run("number of quantiles is larger than num input vals", func(t *testing.T) {
		encoder := KBinsDiscretizer{QuantileScaler{Quantiles: []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}}
		encoder.Fit([]float64{1, 2, 3})
		assert.Equal(t, KBinsDiscretizer{QuantileScaler{Quantiles: []float64{1, 2, 3}}}, encoder)
	})

	t.Run("when fit on nil data not zero value", func(t *testing.T) {
		encoder := KBinsDiscretizer{}
		encoder.Fit(nil)
		assert.Equal(t, KBinsDiscretizer{}, encoder)
	})
}
