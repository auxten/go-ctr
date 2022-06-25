package feature

import "math"

// SampleNormalizerL1 transforms features for single sample to have norm L1=1
type SampleNormalizerL1 struct{}

// Fit is empty, kept only to keep same interface
func (t *SampleNormalizerL1) Fit(_ []float64) {}

// Transform returns L1 normalized vector
func (t *SampleNormalizerL1) Transform(vs []float64) []float64 {
	if t == nil || vs == nil {
		return nil
	}
	vsnorm := make([]float64, len(vs))
	t.TransformInplace(vsnorm, vs)
	return vsnorm
}

// TransformInplace returns L1 normalized vector, inplace
func (t *SampleNormalizerL1) TransformInplace(dest []float64, vs []float64) {
	if t == nil || vs == nil || dest == nil || len(dest) != len(vs) {
		return
	}

	sum := 0.
	for _, v := range vs {
		sum += math.Abs(v)
	}

	for i := range dest {
		if sum == 0 {
			dest[i] = 0
		} else {
			dest[i] = vs[i] / sum
		}
	}
}

// SampleNormalizerL2 transforms features for single sample to have norm L2=1
type SampleNormalizerL2 struct{}

// Fit is empty, kept only to keep same interface
func (t *SampleNormalizerL2) Fit(_ []float64) {}

// Transform returns L2 normalized vector
func (t *SampleNormalizerL2) Transform(vs []float64) []float64 {
	if t == nil || vs == nil {
		return nil
	}
	vsnorm := make([]float64, len(vs))
	t.TransformInplace(vsnorm, vs)
	return vsnorm
}

// TransformInplace returns L2 normalized vector, inplace
func (t *SampleNormalizerL2) TransformInplace(dest []float64, vs []float64) {
	if t == nil || vs == nil || dest == nil || len(dest) != len(vs) {
		return
	}

	sum := 0.
	for _, v := range vs {
		sum += v * v
	}
	sum = math.Sqrt(sum)

	for i := range dest {
		if sum == 0 {
			dest[i] = 0
		} else {
			dest[i] = vs[i] / sum
		}
	}
}
