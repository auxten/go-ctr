package feature_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOneHotEncoderFit(t *testing.T) {
	samples := []struct {
		name   string
		input  []string
		output map[string]uint
		n      int
	}{
		{"basic", []string{"a", "b", "a", "a", "a"}, map[string]uint{"a": 0, "b": 1}, 2},
		{"empty", []string{}, nil, 0},
		{"nil", nil, nil, 0},
		{"same_string", []string{"a", "a", "a"}, map[string]uint{"a": 0}, 1},
		{"empty_string", []string{"", "", ""}, map[string]uint{}, 0},
		{"zeros_single", []string{""}, map[string]uint{}, 0},
		{"single", []string{"a"}, map[string]uint{"a": 0}, 1},
	}
	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OneHotEncoder{}
			encoder.Fit(s.input)
			assert.Equal(t, OneHotEncoder{Mapping: s.output}, encoder)
			assert.Equal(t, s.n, encoder.NumFeatures())
		})
	}
}

func TestOneHotEncoderTransform(t *testing.T) {
	samples := []struct {
		name    string
		mapping map[string]uint
		input   string
		output  []float64
	}{
		{"basic", map[string]uint{"a": 0, "b": 1}, "a", []float64{1, 0}},
		{"basic", map[string]uint{"a": 0, "b": 1}, "b", []float64{0, 1}},
		{"none", map[string]uint{"a": 0, "b": 1}, "c", []float64{0, 0}},
		{"empty_input", map[string]uint{"a": 0, "b": 1}, "", []float64{0, 0}},
		{"empty_vals", nil, "a", nil},
		{"nil_vals", nil, "a", nil},
		{"zeros_single", map[string]uint{"": 0}, "", []float64{1}},
		{"single", map[string]uint{"a": 0}, "a", []float64{1}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OneHotEncoder{Mapping: s.mapping}
			assert.Equal(t, s.output, encoder.Transform(s.input))
		})

		if len(s.output) > 0 {
			t.Run(s.name+"_inplace", func(t *testing.T) {
				encoder := OneHotEncoder{Mapping: s.mapping}
				assert.Equal(t, s.output, encoder.Transform(s.input))

				features := make([]float64, encoder.NumFeatures())
				encoder.TransformInplace(features, s.input)
				assert.Equal(t, s.output, features)

				features = make([]float64, encoder.NumFeatures()+100)
				features[0] = 11223344556677
				features[1] = 10101010110101
				features[99] = 12312312312312

				expected := make([]float64, len(features))
				copy(expected, features)
				copy(expected[10:], s.output)

				encoder.TransformInplace(features[10:10+encoder.NumFeatures()], s.input)
				assert.Equal(t, expected, features)
			})
		}
	}

	t.Run("inplace does not compute when input is wrong", func(t *testing.T) {
		encoder := OneHotEncoder{Mapping: map[string]uint{"a": 0, "b": 1}}
		features := []float64{1.1, 2.1, 3.1, 4.1}
		encoder.TransformInplace(features, "a")
		assert.Equal(t, []float64{1.1, 2.1, 3.1, 4.1}, features)
	})

	t.Run("transform when encoder is nil", func(t *testing.T) {
		var encoder *OneHotEncoder
		assert.Equal(t, []float64(nil), encoder.Transform("abcd"))
	})
}

func TestOneHotEncoderFeatureNames(t *testing.T) {
	t.Run("feature names on empty transformer", func(t *testing.T) {
		var encoder *OneHotEncoder
		assert.Equal(t, []string(nil), encoder.FeatureNames())
	})

	t.Run("feature names", func(t *testing.T) {
		encoder := OneHotEncoder{Mapping: map[string]uint{"a": 0, "b": 1}}
		assert.Equal(t, []string{"a", "b"}, encoder.FeatureNames())
	})
}

func TestOrdinalEncoderFit(t *testing.T) {
	samples := []struct {
		name   string
		input  []string
		output map[string]uint
	}{
		{"basic", []string{"a", "b", "a", "a", "a"}, map[string]uint{"a": 1, "b": 2}},
		{"empty", []string{}, nil},
		{"nil", nil, nil},
		{"same_string", []string{"a", "a", "a"}, map[string]uint{"a": 1}},
		{"empty_string", []string{"", "", ""}, map[string]uint{}},
		{"zeros_single", []string{""}, map[string]uint{}},
		{"single", []string{"a"}, map[string]uint{"a": 1}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OrdinalEncoder{}
			encoder.Fit(s.input)
			assert.Equal(t, OrdinalEncoder{Mapping: s.output}, encoder)
		})
	}
}

func TestOrdinalEncoderTransform(t *testing.T) {
	samples := []struct {
		name   string
		vals   map[string]uint
		input  string
		output float64
	}{
		{"basic", map[string]uint{"a": 1, "b": 3}, "a", 1},
		{"basic", map[string]uint{"a": 1, "b": 3}, "b", 3},
		{"none", map[string]uint{"a": 1, "b": 3}, "c", 0},
		{"empty_input", map[string]uint{"a": 1, "b": 3}, "", 0},
		{"empty_vals", map[string]uint{}, "a", 0},
		{"nil_vals", nil, "a", 0},
		{"zero_single", map[string]uint{"": 1}, "", 1},
		{"single", map[string]uint{"a": 1}, "a", 1},
	}
	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := OrdinalEncoder{Mapping: s.vals}
			assert.Equal(t, s.output, encoder.Transform(s.input))
		})
	}

	t.Run("transform when encoder is nil", func(t *testing.T) {
		var encoder *OrdinalEncoder
		assert.Equal(t, 0., encoder.Transform("abcd"))
	})
}
