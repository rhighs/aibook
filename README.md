# Notes, code snippets and scripts from "Machine Learning with PyTorch and Scikit-Learn" by Sebastian Raschka

- [Official remote url by the author](https://github.com/rasbt/machine-learning-book)
- [Book listing](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-scikit-learn-ebook-dp-B09NW48MR1/dp/B09NW48MR1)

Here are my personal notes, scripts and code snippets I've collected in the course of reading this book. The publication puts a lot of emphasis on the practical side of machine learning while attempting to cover the theory and stat fundamentals behind machine learning as deeply as possible. A great reading for anyone interested in getting started with machine learning engineering. Below I provide an overview of the chapters as per the book's table of contents linking each specific python file.

NOTE: No python notebooks have been used in here seeing them as unnecessary and an annyoing complication in terms of setup. Instead, I consider a python venv + the use of sys args enough to run the scripts and see what they do. If you really can't do without, refer the author's repository.

## Setup

```bash
$ python -m venv venv && source ./venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Usage
```bash
$ python ./ch<2..=19>/main.py <function_to_execute_from_file>
```

> ```bash
>  $ python ./ch12/main.py torch_datasets
> ```
>

## Overview

- ### [Chapter 1: Giving Computers the Ability to Learn from Data](./ch1/main.py)
  - Introduction to supervised, unsupervised, and reinforcement learning

- ### [Chapter 2: Training Simple Machine Learning Algorithms for Classification](./ch2/main.py)
  - Implementing and training a perceptron on the Iris dataset
  - Adaptive linear neurons and gradient descent

- ### [Chapter 3: A Tour of Machine Learning Classifiers Using Scikit-Learn](./ch3/main.py)
  - Deep dive into logistic regression, SVMs, decision trees, and K-nearest neighbors

- ### [Chapter 4: Building Good Training Datasets – Data Preprocessing](./ch4/main.py)
  - Handling missing data, categorical data, and feature scaling
  - Techniques for feature selection and importance

- ### [Chapter 5: Compressing Data via Dimensionality Reduction](./ch5/main.py)
  - Unsupervised and supervised data compression methods
  - Principal component analysis and linear discriminant analysis

- ### [Chapter 6: Learning Best Practices for Model Evaluation and Hyperparameter Tuning](./ch6/main.py)
  - Streamlining workflows with pipelines
  - Using cross-validation for model evaluation
  - Advanced techniques for tuning machine learning models

- ### [Chapter 7: Combining Different Models for Ensemble Learning](./ch7/main.py)
  - Techniques for combining classifiers such as bagging, boosting, and ensemble methods

- ### [Chapter 8: Applying Machine Learning to Sentiment Analysis](./ch8/main.py)
  - Preparing text data and training models for sentiment analysis

- ### [Chapter 9: Predicting Continuous Target Variables with Regression Analysis](./ch9/main.py)
  - Techniques and challenges in linear and polynomial regression
  - Using decision tree and random forest for regression

- ### [Chapter 10: Working with Unlabeled Data – Clustering Analysis](./ch10/main.py)
  - K-means clustering, hierarchical clustering, and DBSCAN
  - Evaluating clustering quality

- ### [Chapter 11: Implementing a Multilayer Artificial Neural Network from Scratch](./ch11/main.py)
  - Basics of neural network architecture and training
  - Implementing a multilayer perceptron for classifying handwritten digits

- ### [Chapter 12: Parallelizing Neural Network Training with PyTorch](./ch12/main.py)
  - Introduction to PyTorch and its functionalities for neural network training

- ### [Chapter 13: Going Deeper – The Mechanics of PyTorch](./ch13/main.py)
  - Deep dive into PyTorch's computation graphs and automatic differentiation

  TODO

- ### [Chapter 14: Classifying Images with Deep Convolutional Neural Networks](./ch14/main.py)
  - Building and training CNNs for image classification with PyTorch

  TODO

- ### [Chapter 15: Modeling Sequential Data Using Recurrent Neural Networks](./ch15/main.py)
  - Implementing RNNs for tasks like sentiment analysis and sequence modeling

  TODO

- ### [Chapter 16: Transformers – Improving Natural Language Processing with Attention Mechanisms](./ch16/main.py)
  - Understanding and implementing transformers for NLP tasks

  TODO

- ### [Chapter 17: Generative Adversarial Networks for Synthesizing New Data](./ch17/main.py)
  - Fundamentals of GANs and their applications in generating synthetic data

  TODO

- ### [Chapter 18: Graph Neural Networks for Capturing Dependencies in Graph Structured Data](./ch18/main.py)
  - Basics and applications of graph neural networks

  TODO

- ### [Chapter 19: Reinforcement Learning for Decision Making in Complex Environments](./ch19/main.py)
  - Introduction to reinforcement learning and its methodologies for complex decision-making tasks

  TODO
