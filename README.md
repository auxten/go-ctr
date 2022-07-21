# edgeRec

All in one Recommendation System can run on small server or edge device (Android/iOS/IoT device etc.)

To create a deep learning based recommendation system, you need to follow the steps below:

1. Implement the `recommend.RecSys` interface including func below:
    ```
   GetUserFeature(int) Tensor
   GetItemFeature(int) Tensor
   SampleGenerator() (<-chan Sample, error)
   ```
2. Call the functions to `Train` and `StartHttpApi`

     ```
    model, _ = rcmd.Train(recSys)
    rcmd.StartHttpApi(model, "/api/v1/recommend", ":8080")
    ```
   a Movie Lens based example is provided in the `example/movielens` directory. Corresponding database structure is in the `example/movielens/data.go` file.

3. If you want better AUC, you can implement the `recommend.`

# Features

- [x] Pure Golang implementation
- [ ] Parameter Server based Online Learning
- [x] Training & Inference all in one binary powered by golang
- Databases support
  - [x] MySQL support
  - [x] SQLite support
  - [ ] Database Aggregation accelerated Feature Normalization
- Feature Engineering
  - [x] Item2vec embedding
  - [ ] Rule based FE config
  - [ ] DeepL based Auto Feature Engineering
- Demo
  - [x] MovieLens Demo 
  - [ ] Android demo
  - [ ] iOS demo

# Benchmark

## Embedding

- Apple M1 Max
- Database: SQLite3
- Model: SkipGram, Optimizer: HierarchicalSoftmax
- WindowSize: 5
- Data: [MovieLens 10m](https://grouplens.org/datasets/movielens/10m/)
```
read 9520886 words 12.169282375s
trained 9519544 words 17.155356791s
```

# Thanks

To make this project work, quite a lot of code are copied and modified from the following libraries:
- Neural Network & Parameter Server: 
  - [go-deep](https://github.com/patrikeh/go-deep)
  - [gorgonia](https://github.com/gorgonia/gorgonia)
  - [pa-m/sklearn](https://github.com/pa-m/sklearn)
- Feature Engineering:
  - [go-featureprocessing](https://github.com/nikolaydubina/go-featureprocessing)
  - [featuremill](https://github.com/dustin-decker/featuremill)
  - [wego](https://github.com/ynqa/wego)
- FastAPI like framework:
  - [go-fastapi](https://github.com/sashabaranov/go-fastapi)

# Papers related

- [Document Embedding with Paragraph Vectors](https://arxiv.org/abs/1507.07998)
- [EdgeRec: Recommender System on Edge in Mobile Taobao](https://arxiv.org/abs/2005.08416) // not very identical implementation