package main

import (
	"database/sql"
)

const (
	DbPath = "movielens.db"
)

var (
	db *sql.DB
)

type UserFeature struct {
	UserId     int       `json:"userId"`
	Embeddings []float64 `json:"embeddings"`
}

type ItemFeature struct {
	MovieId    int       `json:"movieId"`
	Embeddings []float64 `json:"embeddings"`
}

func init() {
	var err error
	db, err = sql.Open("sqlite3", DbPath)
	if err != nil {
		panic(err)
	}
}

func GetUserFeatures(userId int) (uf *UserFeature, err error) {
	var embeddings []float64
	err = db.QueryRow("SELECT * FROM user_features WHERE userId = ?", userId).Scan(&embeddings)
	if err != nil {
		return
	}
	uf = &UserFeature{
		UserId:     userId,
		Embeddings: embeddings,
	}
	return
}
