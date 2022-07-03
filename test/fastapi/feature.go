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
	UserId int `json:"userId"`
}

func init() {
	var err error
	db, err = sql.Open("sqlite3", DbPath)
	if err != nil {
		panic(err)
	}
}

func GetUserFeatures(userId int) []float64 {
	return []float64{}
}
