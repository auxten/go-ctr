package feature

import (
	"math"
	"sort"
)

// Identity is a transformer that returns unmodified input value
type Identity struct{}

// Fit is not used, it is here only to keep same interface as rest of transformers
func (t *Identity) Fit(_ []float64) {}

// Transform returns same value as input
func (t *Identity) Transform(v float64) float64 {
	return v
}

// MinMaxScaler is a transformer that rescales value into range between min and max
type MinMaxScaler struct {
	Min float64
	Max float64
}

// Fit findx min and max value in range
func (t *MinMaxScaler) Fit(vals []float64) {
	for i, v := range vals {
		if i == 0 {
			t.Min = v
			t.Max = v
		}
		if v < t.Min {
			t.Min = v
		}
		if v > t.Max {
			t.Max = v
		}
	}
}

// Transform scales value from 0 to 1 linearly
func (t *MinMaxScaler) Transform(v float64) float64 {
	if t.Min == t.Max {
		return 0
	}
	if v < t.Min {
		return 0.
	}
	if v > t.Max {
		return 1.
	}
	return (v - t.Min) / (t.Max - t.Min)
}

// MaxAbsScaler transforms value into -1 to +1 range linearly
type MaxAbsScaler struct {
	Max float64
}

// Fit finds maximum abssolute value
func (t *MaxAbsScaler) Fit(vals []float64) {
	for i, v := range vals {
		if i == 0 {
			t.Max = v
		}
		if math.Abs(v) > t.Max {
			t.Max = math.Abs(v)
		}
	}
}

// Transform scales value into -1 to +1 range
func (t *MaxAbsScaler) Transform(v float64) float64 {
	if t.Max == 0 {
		return 0
	}
	if v > math.Abs(t.Max) {
		return 1.
	}
	if v < -math.Abs(t.Max) {
		return -1.
	}
	return v / math.Abs(t.Max)
}

// StandardScaler transforms feature into normal standard distribution.
type StandardScaler struct {
	Mean float64
	STD  float64
}

// Fit computes mean and standard deviation
func (t *StandardScaler) Fit(vals []float64) {
	sum := 0.
	for _, v := range vals {
		sum += v
	}
	if len(vals) > 0 {
		t.Mean = sum / float64(len(vals))
		t.STD = std(vals, t.Mean)
	}
}

// Transform centralizes and scales based on standard deviation and mean
func (t *StandardScaler) Transform(v float64) float64 {
	return (v - t.Mean) / t.STD
}

// QuantileScaler transforms any distribution to uniform distribution
// This is done by mapping values to quantiles they belong to.
type QuantileScaler struct {
	Quantiles []float64
}

// Fit sets parameters for quantiles based on input.
// Number of quantiles are specified by size of Quantiles slice.
// If it is empty or nil, then 100 is used as default.
// If input is smaller than number of quantiles, then using length of input.
func (t *QuantileScaler) Fit(vals []float64) {
	if len(vals) == 0 {
		return
	}
	if len(t.Quantiles) == 0 {
		t.Quantiles = make([]float64, 100)
	}
	if len(vals) < len(t.Quantiles) {
		t.Quantiles = t.Quantiles[:len(vals)]
	}

	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)

	f := float64(len(sorted)) / float64(len(t.Quantiles))
	for i := range t.Quantiles {
		idx := int(float64(i) * f)
		t.Quantiles[i] = sorted[idx]
	}
}

// Transform changes distribution into uniform one from 0 to 1
func (t *QuantileScaler) Transform(v float64) float64 {
	if t == nil || len(t.Quantiles) == 0 {
		return 0
	}
	i := sort.SearchFloat64s(t.Quantiles[:], v)
	if i >= len(t.Quantiles) {
		return 1.
	}
	return float64(i+1) / float64(len(t.Quantiles))
}
