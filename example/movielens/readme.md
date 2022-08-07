# MovieLens Example

Original Data: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

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

## The DIN way to split dataset

There is another way to split the MovieLens-20m dataset with userId that is described 
in the [Deep Interest Network](https://arxiv.org/abs/1706.06978) paper.

[MovieLens 20m](https://grouplens.org/datasets/movielens/20m/)

Related SQL:
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

-- import data from csv, do it with any tool

select count(distinct userId) from ratings; -- 138,493 users

create table user as select distinct userId, 0 as is_train  from ratings;

-- choose 100000 random user as train user
update user
set is_train = 1
where userId in
      (SELECT userId
       FROM (select distinct userId from ratings)
       ORDER BY RANDOM()
       LIMIT 100000);

select count(*) from user where is_train != 1; -- 38,493 test users

-- split train and test set of movielens-20m ratings
create table ratings_train as
select r.userId, movieId, rating, timestamp
from ratings r
         left join user u on r.userId = u.userId
where is_train = 1;
create table ratings_test as
select r.userId, movieId, rating, timestamp
from ratings r
         left join user u on r.userId = u.userId
where is_train = 0;

select count(*) from ratings_train; --14,393,526
select count(*) from ratings_test;  --5,606,737
select count(*) from ratings;       --20,000,263
```