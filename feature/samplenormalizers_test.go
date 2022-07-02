package feature

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSampleNormalizserL1(t *testing.T) {
	samples := []struct {
		name   string
		input  []float64
		output []float64
	}{
		{"basic", []float64{1, 2, 3, 4}, []float64{0.1, 0.2, 0.3, 0.4}},
		{"empty", []float64{}, []float64{}},
		{"nil", nil, nil},
		{"zeros", []float64{0, 0, 0}, []float64{0, 0, 0}},
		{"zeros_single", []float64{0}, []float64{0}},
		{"single", []float64{5}, []float64{1}},
		{"single_negative", []float64{-5}, []float64{-1}},
		{"negative", []float64{1, 2, 3, -4}, []float64{0.1, 0.2, 0.3, -0.4}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := SampleNormalizerL1{}
			features := encoder.Transform((s.input))
			assert.Equal(t, s.output, features)

			// inplace
			if len(s.output) > 0 {
				features := make([]float64, len(s.input))
				encoder.TransformInplace(features, s.input)
				assert.Equal(t, s.output, features)

				features = make([]float64, len(s.input)+100)
				features[0] = 11223344556677
				features[1] = 10101010110101
				features[99] = 223312112233
				copy(features[10:], s.output)
				expected := make([]float64, len(features))
				copy(expected, features)

				encoder.TransformInplace(features[10:10+len(s.input)], s.input)
				assert.Equal(t, expected, features)
			}
		})
	}

	t.Run("fit", func(t *testing.T) {
		encoder := SampleNormalizerL1{}
		encoder.Fit(nil)
		assert.Equal(t, SampleNormalizerL1{}, encoder)
	})

	t.Run("inplace does not run when input mismatches", func(t *testing.T) {
		encoder := SampleNormalizerL1{}
		f := []float64{1, 2}
		encoder.TransformInplace(f, []float64{1, 2, 3, 4})
		assert.Equal(t, []float64{1, 2}, f)
	})
}

func TestSampleNormalizserL2(t *testing.T) {
	samples := []struct {
		name   string
		input  []float64
		output []float64
	}{
		{"basic", []float64{1, 1, 3, 5, 8}, []float64{0.1, 0.1, 0.3, 0.5, 0.8}},
		{"empty", []float64{}, []float64{}},
		{"nil", nil, nil},
		{"zeros", []float64{0, 0, 0}, []float64{0, 0, 0}},
		{"zeros_single", []float64{0}, []float64{0}},
		{"single", []float64{5}, []float64{1}},
		{"single_negative", []float64{-5}, []float64{-1}},
		{"basic", []float64{1, 1, -3, 5, -8}, []float64{0.1, 0.1, -0.3, 0.5, -0.8}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := SampleNormalizerL2{}
			features := encoder.Transform((s.input))
			assert.Equal(t, s.output, features)
		})

		if len(s.output) > 0 {
			t.Run(s.name+"_inplace", func(t *testing.T) {
				encoder := SampleNormalizerL2{}

				features := make([]float64, len(s.input))
				encoder.TransformInplace(features, s.input)
				assert.Equal(t, s.output, features)

				features = make([]float64, len(s.input)+100)
				features[0] = 1
				features[1] = 2
				features[10] = 12312 // has to overwrite this
				features[99] = 5

				expected := make([]float64, len(features))
				copy(expected, features)
				copy(expected[10:], s.output)

				encoder.TransformInplace(features[10:10+len(s.input)], s.input)
				assert.Equal(t, expected, features)
			})
		}
	}

	t.Run("fit", func(t *testing.T) {
		encoder := SampleNormalizerL2{}
		encoder.Fit(nil)
		assert.Equal(t, SampleNormalizerL2{}, encoder)
	})

	t.Run("inplace does not run when input mismatches", func(t *testing.T) {
		encoder := SampleNormalizerL2{}
		f := []float64{1, 2}
		encoder.TransformInplace(f, []float64{1, 2, 3, 4})
		assert.Equal(t, []float64{1, 2}, f)
	})
}
