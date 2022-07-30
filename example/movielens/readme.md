# MovieLens Example

Original Data: [MovieLens 100k](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

SQLite DB file:
[movielens.db.zip](https://github.com/auxten/edgeRec/files/9176009/movielens.db.zip)

To run the tests, you need download the SQLite DB file and put it in the current directory.

```shell
# download and unzip the SQLite DB file
wget https://github.com/auxten/edgeRec/files/9176009/movielens.db.zip && unzip movielens.db.zip
```


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