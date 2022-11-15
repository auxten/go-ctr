package utils

import (
	"encoding/binary"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/auxten/go-ctr/nn/metrics"
	"gonum.org/v1/gonum/mat"
)

func ConcatSlice(slices ...[]float64) []float64 {
	result := make([]float64, 0)
	for _, slice := range slices {
		result = append(result, slice...)
	}
	return result
}

func ConcatSlice32(slices ...[]float32) []float32 {
	result := make([]float32, 0)
	for _, slice := range slices {
		result = append(result, slice...)
	}
	return result
}

func Float64toBytes(f float64) []byte {
	bits := math.Float64bits(f)
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, bits)
	return bytes
}

type KeyCnt struct {
	Key string
	Cnt int
}

func TopNOccurrences(s []string, n int) []KeyCnt {
	m := make(map[string]int)
	for _, str := range s {
		if _, ok := m[str]; ok {
			m[str]++
		} else {
			m[str] = 1
		}
	}

	l1 := make([]KeyCnt, 0, len(m))
	for k, v := range m {
		l1 = append(l1, KeyCnt{k, v})
	}

	sort.Slice(l1, func(i, j int) bool {
		if l1[i].Cnt == l1[j].Cnt {
			return l1[i].Key < l1[j].Key
		} else {
			return l1[i].Cnt > l1[j].Cnt
		}
	})
	if len(l1) < n {
		return l1
	}

	return l1[:n]
}

func ParseInt64Seq(s string) []int64 {
	var (
		seq []int64
	)
	for _, str := range strings.Split(s, ",") {
		if str == "" {
			continue
		}
		i, err := strconv.ParseInt(str, 10, 64)
		if err != nil {
			panic(err)
		}
		seq = append(seq, i)
	}
	return seq
}

func Int64SeqToIntSeq(seq []int64) []int {
	result := make([]int, len(seq))
	for i, v := range seq {
		result[i] = int(v)
	}
	return result
}

func Accuracy(prediction, y []float64) float64 {
	var ok float64
	for i := 0; i < len(prediction); i++ {
		if math.Round(float64(prediction[i]-y[i])) == 0 {
			ok += 1.0
		}
	}
	return ok / float64(len(y))
}

func Accuracy32(prediction, y []float32) float32 {
	var ok float32
	for i := 0; i < len(prediction); i++ {
		if math.Round(float64(prediction[i]-y[i])) == 0 {
			ok += 1.0
		}
	}
	return ok / float32(len(y))
}

func RocAuc(pred, y []float64) float64 {
	boolY := make([]float64, len(y))
	for i := 0; i < len(y); i++ {
		if y[i] > 0.5 {
			boolY[i] = 1.0
		} else {
			boolY[i] = 0.0
		}
	}
	yTrue := mat.NewDense(len(y), 1, boolY)
	yScore := mat.NewDense(len(pred), 1, pred)

	return metrics.ROCAUCScore(yTrue, yScore, "", nil)
}

func RocAuc32(pred, y []float32) float32 {
	boolY := make([]float64, len(y))
	for i := 0; i < len(y); i++ {
		if y[i] > 0.5 {
			boolY[i] = 1.0
		} else {
			boolY[i] = 0.0
		}
	}
	pred64 := make([]float64, len(pred))
	for i := 0; i < len(pred); i++ {
		pred64[i] = float64(pred[i])
	}
	yTrue := mat.NewDense(len(y), 1, boolY)
	yScore := mat.NewDense(len(pred), 1, pred64)

	return float32(metrics.ROCAUCScore(yTrue, yScore, "", nil))
}
