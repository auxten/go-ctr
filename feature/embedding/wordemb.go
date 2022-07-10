package embedding

import (
	"github.com/auxten/edgeRec/feature/embedding/model"
	"github.com/auxten/edgeRec/feature/embedding/model/word2vec"
	log "github.com/sirupsen/logrus"
)

func TrainEmbedding(inputCh <-chan string, window int, dim int, iter int) (mod model.Model, err error) {
	if mod, err = word2vec.New(
		word2vec.Window(window),
		word2vec.Dim(dim),
		word2vec.Model(word2vec.SkipGram),
		word2vec.Optimizer(word2vec.HierarchicalSoftmax),
		word2vec.Verbose(),
		word2vec.Iter(iter),
		word2vec.DocInMemory(),
	); err != nil {
		return
	}

	if err = mod.Train(inputCh); err != nil {
		log.Errorf("failed to train embedding: %v", err)
		return
	}
	if err = mod.EmbeddingMap(); err != nil {
		log.Errorf("failed to get embedding map: %v", err)
		return
	}

	return
}
