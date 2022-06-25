package feature

import "sort"

// KBinsDiscretizer based on quantile strategy
type KBinsDiscretizer struct {
	QuantileScaler
}

// Fit fits quantile scaler
func (t *KBinsDiscretizer) Fit(vals []float64) {
	t.QuantileScaler.Fit(vals)
}

// Transform finds index of matched quantile for input
func (t *KBinsDiscretizer) Transform(v float64) float64 {
	if len(t.QuantileScaler.Quantiles) == 0 {
		return 0
	}
	i := sort.SearchFloat64s(t.Quantiles[:], v)
	if i >= len(t.Quantiles) {
		return float64(len(t.Quantiles)) + 1
	}
	return float64(i) + 1
}
