package main

/*
	create table movies
	(
		movieId INTEGER,
		title   TEXT,
		genres  TEXT
	);
*/
type Movie struct {
	MovieId int    `json:"movieId"`
	Title   string `json:"title"`
	Genres  string `json:"genres"`
}

/*
	create table ratings
	(
		userId INTEGER,
		movieId INTEGER,
		rating INTEGER,
		timestamp INTEGER
	);
*/
type Rating struct {
	UserId    int `json:"userId"`
	MovieId   int `json:"movieId"`
	Rating    int `json:"rating"`
	Timestamp int `json:"timestamp"`
}

/*
	create table tags
	(
		userId    INTEGER,
		movieId   INTEGER,
		tag       TEXT,
		timestamp INTEGER
	);
*/
type Tag struct {
	UserId    int    `json:"userId"`
	MovieId   int    `json:"movieId"`
	Tag       string `json:"tag"`
	Timestamp int    `json:"timestamp"`
}
