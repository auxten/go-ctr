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
