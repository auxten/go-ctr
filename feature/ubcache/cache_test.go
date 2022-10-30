package ubcache

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestUbCache(t *testing.T) {
	Convey("test seq filter", t, func() {
		seq := &TimeSeq{
			Ts:    []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
			Items: []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		}
		fs := seq.Filter(0, 0)
		So(fs, ShouldResemble, seq)

		fs = seq.Filter(0, 5)
		So(fs, ShouldResemble, &TimeSeq{
			Ts:    []int64{10, 9, 8, 7, 6},
			Items: []int64{10, 9, 8, 7, 6},
		})

		fs = seq.Filter(0, 15)
		So(fs, ShouldResemble, seq)

		fs = seq.Filter(5, 0)
		So(fs, ShouldResemble, &TimeSeq{
			Ts:    []int64{5, 4, 3, 2, 1},
			Items: []int64{5, 4, 3, 2, 1},
		})

		fs = seq.Filter(5, 3)
		So(fs, ShouldResemble, &TimeSeq{
			Ts:    []int64{5, 4, 3},
			Items: []int64{5, 4, 3},
		})
	})

	Convey("test ub cache", t, func() {
		ubc := NewUserBehaviorCache()
		ubc.Set(1, &TimeSeq{
			Ts:    []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
			Items: []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		})
		gs, err := ubc.Get(1, 0, 0)
		So(err, ShouldBeNil)
		So(gs, ShouldResemble, &TimeSeq{
			Ts:    []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
			Items: []int64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		})

		gs, err = ubc.Get(1, 0, 5)
		So(err, ShouldBeNil)
		So(gs, ShouldResemble, &TimeSeq{
			Ts:    []int64{10, 9, 8, 7, 6},
			Items: []int64{10, 9, 8, 7, 6},
		})

		ubc.Set(2, &TimeSeq{
			Ts:    []int64{10, 9, 8, 7, 6},
			Items: []int64{10, 9, 8, 7, 6},
		})
		gs, err = ubc.Get(2, 0, 0)
		So(err, ShouldBeNil)
		So(gs, ShouldResemble, &TimeSeq{
			Ts:    []int64{10, 9, 8, 7, 6},
			Items: []int64{10, 9, 8, 7, 6},
		})

		//not found
		gs, err = ubc.Get(3, 0, 0)
		So(err, ShouldNotBeNil)
		So(gs, ShouldBeNil)

		//delete
		ubc.Delete(2)
		gs, err = ubc.Get(2, 0, 0)
		So(err, ShouldNotBeNil)
		So(gs, ShouldBeNil)

		//batch set
		ubc.BatchSet(map[int64]*TimeSeq{
			1: &TimeSeq{
				Ts:    []int64{10, 9, 8},
				Items: []int64{10, 9, 8},
			},
		})
		gs, err = ubc.Get(1, 0, 0)
		So(err, ShouldBeNil)
		So(gs, ShouldResemble, &TimeSeq{
			Ts:    []int64{10, 9, 8},
			Items: []int64{10, 9, 8},
		})

		//clear
		ubc.Clear()
		gs, err = ubc.Get(1, 0, 0)
		So(err, ShouldNotBeNil)
		So(gs, ShouldBeNil)
	})
}
