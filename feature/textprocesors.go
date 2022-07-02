package feature

import (
	"math"
	"strings"
)

// CountVectorizer performs bag of words encoding of text.
//
// Separator should not be a part of any word.
// Responsibility to ensure this is on caller.
// Words that have separator as its substring will be ommited.
//
// Mapping should contain all values from 0 to N where N is len(Mapping).
// Responsibility to ensure this is on caller.
// If some index is higher than N or lower than 0, then code will panic.
// If some index is not set, then that index will be skipped.
// If some index is set twice, then index will have sum of words.
type CountVectorizer struct {
	Mapping   map[string]uint // word to index
	Separator string          // default space
}

// Fit assigns a number from 0 to N for each word in input, where N is number of words
func (t *CountVectorizer) Fit(vals []string) {
	if t.Separator == "" {
		t.Separator = " "
	}
	if len(vals) == 0 {
		return
	}
	t.Mapping = make(map[string]uint)
	var count uint = 0
	for _, v := range vals {
		ws := strings.Split(v, t.Separator)
		for _, w := range ws {
			if w == "" {
				continue
			}
			if _, ok := t.Mapping[w]; !ok {
				t.Mapping[w] = count
				count++
			}
		}
	}
}

// NumFeatures returns num of features made for single input field
func (t *CountVectorizer) NumFeatures() int {
	if t == nil {
		return 0
	}
	return len(t.Mapping)
}

// Transform counts how many times each word appeared in input
func (t *CountVectorizer) Transform(v string) []float64 {
	if t == nil || v == "" || len(t.Mapping) == 0 {
		return nil
	}
	counts := make([]float64, t.NumFeatures())
	t.TransformInplace(counts, v)
	return counts
}

// FeatureNames returns slice with produced feature names
func (t *CountVectorizer) FeatureNames() []string {
	if t == nil || len(t.Mapping) == 0 {
		return nil
	}

	names := make([]string, t.NumFeatures())
	for w, i := range t.Mapping {
		names[i] = w
	}
	return names
}

// TransformInplace counts how many times each word appeared in input, inplace version.
// It is responsibility of caller to zero-out destination.
// Using zero memory allocation algorithm based on `strings.Split`.
// Utilizing that string is slice of bytes.
// Works fine with UTF-8.
func (t *CountVectorizer) TransformInplace(dest []float64, v string) {
	if t == nil || t.Separator == "" || len(t.Mapping) == 0 || len(dest) != t.NumFeatures() {
		return
	}
	sep := t.Separator

	n := strings.Count(v, sep)
	if n == 0 {
		// no separators, try to match whole string
		if idx, ok := t.Mapping[v]; ok {
			dest[idx] = 1
		}
		return
	}

	j := 0 // looking for position of separator in v starting from here
	for i := 0; i < n; i++ {
		// we are guaranteed to find next separator, m >= 0
		m := strings.Index(v[j:], sep)

		// word between separators
		if idx, ok := t.Mapping[v[j:j+m]]; ok {
			dest[idx]++
		}

		// increment by current word length and separator length
		j += m + len(sep)
	}
	if j != len(v) {
		// if string did not end with separator, it ended with word
		if idx, ok := t.Mapping[v[j:]]; ok {
			dest[idx]++
		}
	}
}

// TFIDFVectorizer performs tf-idf vectorization on top of count vectorization.
// Based on: https://scikit-learn.org/stable/modules/feature_extraction.html
// Using non-smooth version, adding 1 to log instead of denominator in idf.
//
// DocCount should have len of len(CountVectorizer.Mapping).
// It is responsibility of a caller to sensure it is so.
type TFIDFVectorizer struct {
	CountVectorizer
	DocCount     []uint // number of documents where i-th word from CountVectorizer appeared in
	NumDocuments int
	Normalizer   SampleNormalizerL2
}

// Fit fits CountVectorizer and extra information for tf-idf computation
func (t *TFIDFVectorizer) Fit(vals []string) {
	t.CountVectorizer.Fit(vals)
	if len(vals) == 0 {
		return
	}

	t.NumDocuments = len(vals)
	t.DocCount = make([]uint, t.NumFeatures())

	// second pass over whole input to count how many documents each word appeared in
	for _, v := range vals {
		counts := t.CountVectorizer.Transform(v)
		for i, v := range counts {
			if v > 0 {
				t.DocCount[i]++
			}
		}
	}
}

// NumFeatures returns number of features for single field
func (t *TFIDFVectorizer) NumFeatures() int {
	if t == nil {
		return 0
	}
	return len(t.CountVectorizer.Mapping)
}

// Transform performs tf-idf computation
func (t *TFIDFVectorizer) Transform(v string) []float64 {
	if t == nil {
		return nil
	}
	features := make([]float64, t.NumFeatures())
	t.TransformInplace(features, v)
	return features
}

// TransformInplace performs tf-idf computation, inplace.
// It is responsibility of caller to zero-out destination.
func (t *TFIDFVectorizer) TransformInplace(dest []float64, v string) {
	if t == nil || dest == nil || len(dest) != t.NumFeatures() {
		return
	}
	t.CountVectorizer.TransformInplace(dest, v)

	for i, tf := range dest {
		if tf > 0 && t.DocCount[i] > 0 {
			dest[i] = tf * (math.Log(float64(t.NumDocuments)/float64(t.DocCount[i])) + 1)
		} else {
			dest[i] = 0
		}
	}

	t.Normalizer.TransformInplace(dest, dest)
}

// FeatureNames returns slice with produced feature names.
func (t *TFIDFVectorizer) FeatureNames() []string {
	if t == nil {
		return nil
	}
	return t.CountVectorizer.FeatureNames()
}
