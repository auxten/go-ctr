package utils

import (
	"encoding/binary"
	"math"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/integrate"
	"gonum.org/v1/gonum/stat"
)

func ConcatSlice(slices ...[]float64) []float64 {
	result := make([]float64, 0)
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

func RocAuc(label []bool, y []float64) float64 {
	tpr, fpr, _ := stat.ROC(nil, y, label, nil)
	return integrate.Trapezoidal(fpr, tpr)
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
