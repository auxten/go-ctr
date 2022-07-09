package embedding

import (
	"database/sql"
	"fmt"
	"os"
	"testing"

	"github.com/auxten/edgeRec/feature/embedding/embedding"
	"github.com/auxten/edgeRec/feature/embedding/model/modelutil/vector"
	"github.com/auxten/edgeRec/feature/embedding/search"
	_ "github.com/mattn/go-sqlite3"
	log "github.com/sirupsen/logrus"
	. "github.com/smartystreets/goconvey/convey"
)

var (
	SearchMovieIds = []string{"63828", "59315", "58559", "59784"}
	ModelFilePath  = "../../test/model.txt"
)

func TestEmbedding(t *testing.T) {
	log.SetLevel(log.DebugLevel)
	Convey("word embedding", t, func() {
		db, err := sql.Open("sqlite3", "../../test/movielens-10m.db")

		So(err, ShouldBeNil)
		defer db.Close()
		rows, err := db.Query(`select userId, movieId
										from ratings r
										where r.rating > 1
										order by userId, timestamp`)
		So(err, ShouldBeNil)
		defer rows.Close()
		inputCh := make(chan string, 1000)
		go func() {
			for rows.Next() {
				var userId, movieId int
				err := rows.Scan(&userId, &movieId)
				if err != nil {
					t.Fatal(err)
				}
				inputCh <- fmt.Sprintf("%d", movieId)
			}
			close(inputCh)
			log.Debug("input channel closed")
		}()
		mod, err := TrainEmbedding(inputCh, 5)
		modelFileWriter, err := os.OpenFile(ModelFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		So(err, ShouldBeNil)
		defer modelFileWriter.Close()
		err = mod.Save(modelFileWriter, vector.Agg)
		So(err, ShouldBeNil)
		modelFileWriter.Sync()

		modelFileReader, err := os.Open(ModelFilePath)
		embs, err := embedding.Load(modelFileReader)
		So(err, ShouldBeNil)
		searcher, err := search.New(embs...)
		So(err, ShouldBeNil)

		for _, SearchMovieId := range SearchMovieIds {
			neighbors, _ := searcher.SearchInternal(SearchMovieId, 10)
			neighbors.Describe()

			var (
				movieId int
				title   string
				genres  string
			)
			r, _ := db.Query("select movieId, title, genres from movies where movieId = ?", SearchMovieId)
			r.Next()
			r.Scan(&movieId, &title, &genres)
			r.Close()
			fmt.Printf("%d %s %s\n", movieId, title, genres)

			for _, n := range neighbors {
				r, err := db.Query("select movieId, title, genres from movies where movieId = ?", n.Word)
				So(err, ShouldBeNil)
				r.Next()
				r.Scan(&movieId, &title, &genres)
				r.Close()
				fmt.Printf("%d %s %s\n", movieId, title, genres)
			}
		}
	})
}
