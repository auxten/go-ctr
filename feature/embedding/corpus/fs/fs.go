// Copyright © 2020 wego authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fs

import (
	"fmt"
	"strings"

	"github.com/auxten/edgeRec/feature/embedding/corpus"
	"github.com/auxten/edgeRec/feature/embedding/corpus/cpsutil"
	"github.com/auxten/edgeRec/feature/embedding/corpus/dictionary"
	"github.com/auxten/edgeRec/feature/embedding/util/clock"
	"github.com/auxten/edgeRec/feature/embedding/util/verbose"
)

type Corpus struct {
	doc <-chan string

	dic    *dictionary.Dictionary
	maxLen int

	toLower bool
	filters cpsutil.Filters
}

func New(r <-chan string, toLower bool, maxCount, minCount int) corpus.Corpus {
	return &Corpus{
		doc: r,
		dic: dictionary.New(),

		toLower: toLower,
		filters: cpsutil.Filters{
			cpsutil.MaxCount(maxCount),
			cpsutil.MinCount(minCount),
		},
	}
}

func (c *Corpus) IndexedDoc() []int {
	return nil
}

func (c *Corpus) BatchWords(ch chan []int, batchSize int) error {
	cursor, ids := 0, make([]int, batchSize)
	if err := cpsutil.ReadWord(c.doc, func(word string) error {
		if c.toLower {
			word = strings.ToLower(word)
		}

		id, _ := c.dic.ID(word)
		if c.filters.Any(id, c.dic) {
			return nil
		}

		ids[cursor] = id
		cursor++
		if cursor == batchSize {
			ch <- ids
			cursor, ids = 0, make([]int, batchSize)
		}
		return nil
	}); err != nil {
		return err
	}

	// send left words
	ch <- ids[:cursor]
	close(ch)
	return nil
}

func (c *Corpus) Dictionary() *dictionary.Dictionary {
	return c.dic
}

func (c *Corpus) Len() int {
	return c.maxLen
}

func (c *Corpus) Load(verbose *verbose.Verbose, logBatch int) error {
	clk := clock.New()
	if err := cpsutil.ReadWord(c.doc, func(word string) error {
		if c.toLower {
			word = strings.ToLower(word)
		}

		c.dic.Add(word)
		c.maxLen++
		verbose.Do(func() {
			if c.maxLen%logBatch == 0 {
				fmt.Printf("read %d words %v\r", c.maxLen, clk.AllElapsed())
			}
		})

		return nil
	}); err != nil {
		return err
	}
	verbose.Do(func() {
		fmt.Printf("read %d words %v\r\n", c.maxLen, clk.AllElapsed())
	})

	return nil
}
