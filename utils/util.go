package utils

func ConcatSlice(slices ...[]float64) []float64 {
	result := make([]float64, 0)
	for _, slice := range slices {
		result = append(result, slice...)
	}
	return result
}
