package utils

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestUtils(t *testing.T) {
	Convey("test get top 5", t, func() {
		l := []string{"a", "a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "d", "d", "e", "e", "f", "g", "h", "i", "j"}
		top5 := TopNOccurrences(l, 5)
		So(top5, ShouldResemble, []KeyCnt{{"a", 5}, {"b", 4}, {"c", 3}, {"d", 2}, {"e", 2}})
		top3 := TopNOccurrences(l, 3)
		So(top3, ShouldResemble, []KeyCnt{{"a", 5}, {"b", 4}, {"c", 3}})
		top1 := TopNOccurrences(l, 1)
		So(top1, ShouldResemble, []KeyCnt{{"a", 5}})

		top100 := TopNOccurrences(l, 100)
		So(top100, ShouldResemble, []KeyCnt{{"a", 5}, {"b", 4}, {"c", 3}, {"d", 2}, {"e", 2}, {"f", 1}, {"g", 1}, {"h", 1}, {"i", 1}, {"j", 1}})
		So(len(top100), ShouldEqual, 10)
	})
}

func TestGetAUC(t *testing.T) {
	Convey("test auc", t, func() {
		auc := RocAuc([]float64{0.1, 0.35, 0.4, 0.8}, []float64{0, 1, 0, 1})
		So(auc, ShouldEqual, .75)
		auc = RocAuc([]float64{0.1, 0.4, 0.35, 0.8}, []float64{0, 0, 1, 1})
		So(auc, ShouldEqual, .75)
	})
}

func TestParseInt64Seq(t *testing.T) {
	Convey("test parse int64 seq", t, func() {
		seq := ParseInt64Seq("1,2,3,4,5")
		So(seq, ShouldResemble, []int64{1, 2, 3, 4, 5})

		seq = ParseInt64Seq("1,2,3,4,5,")
		So(seq, ShouldResemble, []int64{1, 2, 3, 4, 5})

		// empty string
		seq = ParseInt64Seq("")
		So(seq, ShouldHaveLength, 0)
	})
}

func TestInt64SeqToIntSeq(t *testing.T) {
	Convey("test int64 seq to int seq", t, func() {
		seq := Int64SeqToIntSeq([]int64{1, 2, 3, 4, 5})
		So(seq, ShouldResemble, []int{1, 2, 3, 4, 5})
	})
}
