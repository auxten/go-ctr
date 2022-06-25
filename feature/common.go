package feature

import "math"

func std(vals []float64, mean float64) float64 {
	sum := 0.
	for _, v := range vals {
		sum += math.Abs(v-mean) * math.Abs(v-mean)
	}
	return math.Sqrt(sum / (float64(len(vals)) - 1))
}
