package feature

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCountVectorizer(t *testing.T) {
	samplesFit := []struct {
		name   string
		input  []string
		output map[string]uint
	}{
		{"basic", []string{"a b", "b a", "a", "b", ""}, map[string]uint{"a": 0, "b": 1}},
		{"same_string", []string{"a", "a", "a"}, map[string]uint{"a": 0}},
		{"empty_string", []string{"", "", ""}, map[string]uint{}},
		{"zeros_single", []string{""}, map[string]uint{}},
		{"single", []string{"a"}, map[string]uint{"a": 0}},
		{"empty", nil, nil},
	}

	for _, s := range samplesFit {
		t.Run(s.name, func(t *testing.T) {
			encoder := CountVectorizer{}
			encoder.Fit(s.input)
			assert.Equal(t, CountVectorizer{Mapping: s.output, Separator: " "}, encoder)
		})
	}

	t.Run("num features is zero for nil encoder", func(t *testing.T) {
		var encoder *CountVectorizer
		assert.Equal(t, 0, encoder.NumFeatures())
	})

	t.Run("transform returns nil on nil encoder", func(t *testing.T) {
		var encoder *CountVectorizer
		assert.Equal(t, []float64(nil), encoder.Transform("asdf"))
	})

	t.Run("feature names on empty transformer", func(t *testing.T) {
		var encoder *CountVectorizer
		assert.Equal(t, []string(nil), encoder.FeatureNames())
	})

	t.Run("feature names", func(t *testing.T) {
		encoder := CountVectorizer{Mapping: map[string]uint{"a": 1, "b": 0}}
		assert.Equal(t, []string{"b", "a"}, encoder.FeatureNames())
	})

	samplesTransform := []struct {
		name    string
		sep     string
		mapping map[string]uint
		input   string
		output  []float64
	}{
		{"empty string", "", map[string]uint{"a": 0, "b": 1, "c": 2}, "a b c", []float64{0, 0, 0}},
		{"no separator", " ", map[string]uint{"a": 0, "b": 1, "c": 2}, "a", []float64{1, 0, 0}},
		{"no separator repeating not counted", " ", map[string]uint{"a": 0, "b": 1, "c": 2}, "aaa", []float64{0, 0, 0}},
		{"no separator utf-8", " ", map[string]uint{"안녕": 0, "b": 1, "c": 2}, "안녕", []float64{1, 0, 0}},
		{"no separator utf-8 repeating not counted", " ", map[string]uint{"a": 0, "b": 1, "c": 2}, "안녕안녕안녕", []float64{0, 0, 0}},
		{"basic", " ", map[string]uint{"a": 0, "b": 1, "c": 2}, "a b c", []float64{1, 1, 1}},
		{"ending with separator", " ", map[string]uint{"a": 0, "b": 1, "c": 2}, "a b c ", []float64{1, 1, 1}},
		{"separators continuosly", " ", map[string]uint{"a": 0, "b": 1, "c": 2}, " a b    c  ", []float64{1, 1, 1}},
		{"counting", " ", map[string]uint{"a": 0, "b": 1, "c": 2}, "a a a b b c", []float64{3, 2, 1}},
	}

	for _, s := range samplesTransform {
		t.Run("transform_inplace_"+s.name, func(t *testing.T) {
			tr := CountVectorizer{Separator: s.sep, Mapping: s.mapping}
			assert.Equal(t, s.output, tr.Transform(s.input))
		})
	}
}

func TestTFIDFVectorizerFit(t *testing.T) {
	samples := []struct {
		name        string
		ndocs       int
		doccount    []uint
		mapping     map[string]uint
		input       []string
		numFeatures int
	}{
		{"basic", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, []string{"a a a b b", "a a a c", "a a", "a a a", "a a a a", "a a a c c"}, 3},
		{"empty encoder empty input", 0, []uint(nil), map[string]uint(nil), nil, 0},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := TFIDFVectorizer{}
			expectedEncoder := TFIDFVectorizer{
				CountVectorizer: CountVectorizer{Mapping: s.mapping, Separator: " "},
				NumDocuments:    s.ndocs,
				DocCount:        s.doccount,
			}
			encoder.Fit(s.input)
			assert.Equal(t, expectedEncoder, encoder)
			assert.Equal(t, s.numFeatures, encoder.NumFeatures())
		})
	}

	t.Run("transofmer is nil", func(t *testing.T) {
		var encoder *TFIDFVectorizer
		assert.Equal(t, []float64(nil), encoder.Transform("asdf asdf"))
		assert.Equal(t, 0, encoder.NumFeatures())
	})
}

// test is based on data from: https://scikit-learn.org/stable/modules/feature_extraction.html
func TestTFIDFVectorizerTransform(t *testing.T) {
	samples := []struct {
		name     string
		ndocs    int
		doccount []uint
		mapping  map[string]uint
		input    string
		output   []float64
	}{
		{"basic_1", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, "a a a c", []float64{0.8194099510753755, 0, 0.5732079309279058}},
		{"basic_2", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, "a a", []float64{1, 0, 0}},
		{"basic_3", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, "a a a", []float64{1, 0, 0}},
		{"basic_4", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, "a a a a", []float64{1, 0, 0}},
		{"basic_5", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, "a a a b b", []float64{0.47330339145578754, 0.8808994832762984, 0}},
		{"basic_6", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, "a a a c c", []float64{0.58149260706886, 0, 0.8135516873095773}},
		{"not found", 6, []uint{6, 1, 2}, map[string]uint{"a": 0, "b": 1, "c": 2}, "dddd", []float64{0, 0, 0}},
		{"empty input", 2, []uint{1, 2}, map[string]uint{"a": 0, "b": 1}, "     ", []float64{0, 0}},
		{"empty vals", 2, []uint{1, 2}, map[string]uint{}, " b  a  ", []float64{}},
		{"nil input", 2, []uint{1, 2}, map[string]uint{}, "", []float64{}},
	}

	for _, s := range samples {
		t.Run(s.name, func(t *testing.T) {
			encoder := TFIDFVectorizer{
				CountVectorizer: CountVectorizer{Mapping: s.mapping, Separator: " "},
				NumDocuments:    s.ndocs,
				DocCount:        s.doccount,
			}
			assert.Equal(t, s.output, encoder.Transform(s.input))
		})

		if len(s.output) > 0 {
			t.Run(s.name+"_inplace", func(t *testing.T) {
				encoder := TFIDFVectorizer{
					CountVectorizer: CountVectorizer{Mapping: s.mapping, Separator: " "},
					NumDocuments:    s.ndocs,
					DocCount:        s.doccount,
				}

				features := make([]float64, encoder.NumFeatures())
				encoder.TransformInplace(features, s.input)
				assert.Equal(t, s.output, features)

				// note, values in copied range should be zero
				features = make([]float64, encoder.NumFeatures()+100)
				features[0] = 11223344556677
				features[1] = 10101010110101
				features[99] = 1231231231

				expected := make([]float64, len(features))
				copy(expected, features)
				copy(expected[10:], s.output)

				encoder.TransformInplace(features[10:10+encoder.NumFeatures()], s.input)
				assert.Equal(t, expected, features)
			})
		}
	}

	t.Run("inplace does not run when dest len is not equal num features", func(t *testing.T) {
		encoder := TFIDFVectorizer{
			CountVectorizer: CountVectorizer{Mapping: map[string]uint{"a": 0, "b": 1}, Separator: " "},
			NumDocuments:    5,
			DocCount:        []uint{2, 5},
		}

		features := []float64{1, 2, 3, 4}
		encoder.TransformInplace(features, "a b c d")
		assert.Equal(t, []float64{1, 2, 3, 4}, features)
	})
}

func TestTFIDFVectorizerFeatureNames(t *testing.T) {
	t.Run("feature names on empty transformer", func(t *testing.T) {
		var encoder *TFIDFVectorizer
		assert.Equal(t, []string(nil), encoder.FeatureNames())
	})

	t.Run("feature names", func(t *testing.T) {
		encoder := TFIDFVectorizer{CountVectorizer: CountVectorizer{Mapping: map[string]uint{"a": 1, "b": 0}}}
		assert.Equal(t, []string{"b", "a"}, encoder.FeatureNames())
	})
}
