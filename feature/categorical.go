package feature

// OneHotEncoder encodes string value to corresponding index
//
// Mapping should contain all values from 0 to N where N is len(Mapping).
// Responsibility to ensure this is on caller.
// If some index is higher than N or lower than 0, then code will panic.
// If some index is not set, then that index will be skipped.
// If some index is set twice, then index will have effect of either of words.
type OneHotEncoder struct {
	Mapping map[string]uint // word to index
}

// Fit assigns each value from inputs a number
// based on order of occurrence in input data.
// Ignoring empty strings in input.
func (t *OneHotEncoder) Fit(vs []string) {
	if t == nil || len(vs) == 0 {
		return
	}
	t.Mapping = make(map[string]uint)
	for _, v := range vs {
		if v == "" {
			continue
		}
		if _, ok := t.Mapping[v]; !ok {
			t.Mapping[v] = uint(len(t.Mapping))
		}
	}
}

// NumFeatures returns number of features one field is expanded
func (t *OneHotEncoder) NumFeatures() int {
	return len(t.Mapping)
}

// Transform assigns 1 to value that is found
func (t *OneHotEncoder) Transform(v string) []float64 {
	if t == nil || len(t.Mapping) == 0 {
		return nil
	}
	features := make([]float64, t.NumFeatures())
	t.TransformInplace(features, v)
	return features
}

// TransformInplace assigns 1 to value that is found, inplace.
// It is responsibility of a caller to reset destination to 0.
func (t *OneHotEncoder) TransformInplace(dest []float64, v string) {
	if t == nil || len(t.Mapping) == 0 || len(dest) != t.NumFeatures() {
		return
	}
	if idx, ok := t.Mapping[v]; ok {
		dest[idx] = 1
	}
}

// FeatureNames returns names of each produced value.
func (t *OneHotEncoder) FeatureNames() []string {
	if t == nil || len(t.Mapping) == 0 {
		return nil
	}
	names := make([]string, t.NumFeatures())
	for w, i := range t.Mapping {
		names[i] = w
	}
	return names
}

// OrdinalEncoder returns 0 for string that is not found, or else a number for that string
//
// Mapping should contain all values from 0 to N where N is len(Mapping).
// Responsibility to ensure this is on caller.
// If some index is higher than N or lower than 0, then code will panic.
// If some index is not set, then that index will be skipped.
// If some index is set twice, then index will have effect of either of words.
type OrdinalEncoder struct {
	Mapping map[string]uint
}

// Fit assigns each word value from 1 to N
// Ignoring empty strings in input.
func (t *OrdinalEncoder) Fit(vals []string) {
	if t == nil || len(vals) == 0 {
		return
	}
	t.Mapping = make(map[string]uint)
	for _, v := range vals {
		if v == "" {
			continue
		}
		if _, ok := t.Mapping[v]; !ok {
			t.Mapping[v] = uint(len(t.Mapping) + 1)
		}
	}
}

// Transform returns number of input, if not found returns zero value which is 0
func (t *OrdinalEncoder) Transform(v string) float64 {
	if t == nil {
		return 0
	}
	return float64(t.Mapping[v])
}
