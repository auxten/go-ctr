# edgeRec

All in one Recommendation System running on Edge device (Android/iOS etc..)


# Features

- [x] Parameter Server based Online Learning
- [ ] DeepL based Auto Feature Engineering
- [ ] Database Aggregation accelerated Feature Normalization
- [x] Training & Inference all in one binary powered by golang

# Thanks

To make this project work, quite a lot of code are copied and modified from the following libraries:
- Neural Network & Parameter Server: 
  - [go-deep](https://github.com/patrikeh/go-deep)
  - [gorgonia](https://github.com/gorgonia/gorgonia)
- Feature Engineering:
  - [go-featureprocessing](https://github.com/nikolaydubina/go-featureprocessing)
  - [featuremill](https://github.com/dustin-decker/featuremill)

# Papers related

- [EdgeRec: Recommender System on Edge in Mobile Taobao](https://arxiv.org/abs/2005.08416) // not very identical