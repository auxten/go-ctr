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

package memory

import (
	"fmt"
	"strings"

	"github.com/auxten/go-ctr/feature/embedding/corpus"
	"github.com/auxten/go-ctr/feature/embedding/corpus/cpsutil"
	"github.com/auxten/go-ctr/feature/embedding/corpus/dictionary"
	"github.com/auxten/go-ctr/feature/embedding/util/clock"
	"github.com/auxten/go-ctr/feature/embedding/util/verbose"
)

type Corpus struct {
	doc <-chan string

	dic    *dictionary.Dictionary
	maxLen int
	idoc   []int

	toLower bool
	filters cpsutil.Filters
}

func New(doc <-chan string, toLower bool, maxCount, minCount int) corpus.Corpus {
	return &Corpus{
		doc:  doc,
		dic:  dictionary.New(),
		idoc: make([]int, 0),

		toLower: toLower,
		filters: cpsutil.Filters{
			cpsutil.MaxCount(maxCount),
			cpsutil.MinCount(minCount),
		},
	}
}

func (c *Corpus) IndexedDoc() []int {
	var res []int
	for _, id := range c.idoc {
		if c.filters.Any(id, c.dic) {
			continue
		}
		res = append(res, id)
	}
	return res
}

func (c *Corpus) BatchWords(chan []int, int) error {
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
		id, _ := c.dic.ID(word)
		c.maxLen++
		c.idoc = append(c.idoc, id)
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
