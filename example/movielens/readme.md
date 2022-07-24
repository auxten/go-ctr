# MovieLens Example

Original Data: [MovieLens 100k](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

SQLite DB file:
[movielens.db.zip](https://github.com/auxten/edgeRec/files/9176009/movielens.db.zip)

To run the movielens demo just unzip the movielens.db.zip file and put the movielens.db file in the `example` dir.


The table DDL:
```

Table DDL:
```sql
create table movies
(
    movieId INTEGER,
    title   TEXT,
    genres  TEXT
);

create table ratings
(
    userId INTEGER,
    movieId INTEGER,
    rating INTEGER,
    timestamp INTEGER
);

create table tags
(
    userId    INTEGER,
    movieId   INTEGER,
    tag       TEXT,
    timestamp INTEGER
);
```

SQL that split training set and test set by 80% and 20%:
```sql
create table ratings_train as 
    select * from ratings order by timestamp asc limit 80000;
create table ratings_test as 
    select * from ratings order by timestamp asc limit 80000, 100836;
```