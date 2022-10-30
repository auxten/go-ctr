package ubcache

import (
	"fmt"
	"sync"
)

//TimeSeq is a sequence in Ts desc order
type TimeSeq struct {
	Ts    []int64
	Items []int64
}

type UserBehavior map[int64]*TimeSeq //map[userId]*TimeSeq

type UserBehaviorCache struct {
	sync.RWMutex
	ub UserBehavior
}

func NewUserBehaviorCache() *UserBehaviorCache {
	return &UserBehaviorCache{
		ub: make(UserBehavior),
	}
}

//Set user behavior sequence by userId
func (c *UserBehaviorCache) Set(userId int64, seq *TimeSeq) {
	c.Lock()
	defer c.Unlock()
	c.ub[userId] = seq
}

//BatchSet batch set user behavior sequence
func (c *UserBehaviorCache) BatchSet(ub UserBehavior) {
	c.Lock()
	defer c.Unlock()
	for k, v := range ub {
		c.ub[k] = v
	}
}

//Delete user behavior sequence by userId
func (c *UserBehaviorCache) Delete(userId int64) {
	c.Lock()
	defer c.Unlock()
	delete(c.ub, userId)
}

//Clear all user behavior sequence
func (c *UserBehaviorCache) Clear() {
	c.Lock()
	defer c.Unlock()
	c.ub = make(UserBehavior)
}

//Get user behavior sequence by userId and filtered by maxTs and count
func (c *UserBehaviorCache) Get(userId int64, maxTs int64, count int64) (seq *TimeSeq, err error) {
	c.RLock()
	defer c.RUnlock()
	seq, ok := c.ub[userId]
	if !ok {
		return nil, fmt.Errorf("user %d not found", userId)
	}
	seq = seq.Filter(maxTs, count)

	return seq, nil
}

//Filter sequence by maxTs and maxLen
func (seq *TimeSeq) Filter(maxTs int64, maxLen int64) *TimeSeq {
	if maxTs == 0 {
		maxTs = seq.Ts[0]
	}
	count := int(maxLen)
	if count == 0 {
		count = len(seq.Ts)
	}
	var (
		i int
	)
	for i = 0; i < len(seq.Ts); i++ {
		if seq.Ts[i] <= maxTs {
			break
		}
	}
	if i+count > len(seq.Ts) {
		count = len(seq.Ts) - i
	}
	return &TimeSeq{
		Ts:    seq.Ts[i : i+count],
		Items: seq.Items[i : i+count],
	}
}
