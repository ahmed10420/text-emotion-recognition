# Text Emotion Recognition with BiLSTM and DistilBERT

This project focuses on multiclass text emotion recognition using Deep Learning and Transformer-based NLP models.

The goal is to build a system that takes a text sentence as input and predicts the dominant emotion expressed in it.

The model predicts one of six emotion classes:

- sadness
- joy
- love
- anger
- fear
- surprise

The project was developed progressively in three main stages:

1. A baseline model using BiLSTM and GloVe embeddings.
2. An improved model using DistilBERT.
3. A final DistilBERT model enhanced with class weights to handle class imbalance.

The final model is integrated into a Streamlit web interface that allows users to enter a text and get the predicted emotion with confidence scores and probability visualization.

---

## Project Motivation

Text emotion recognition is an important task in Natural Language Processing.  
Unlike simple sentiment analysis, which usually classifies text as positive or negative, emotion recognition aims to identify a more specific emotional state.

For example:

| Text | Predicted Emotion |
|---|---|
| I feel very happy today | joy |
| I am really angry about this situation | anger |
| I feel sad and lonely tonight | sadness |
| I am afraid of tomorrow’s exam | fear |
| I love spending time with my family | love |
| I did not expect this surprise | surprise |

The main objective of this project is to classify short text sentences into six emotion categories using Deep Learning and Transformer-based models.

---

## Problem Statement

The main problem addressed in this project is:

> How can we automatically identify the dominant emotion expressed in a short text using Deep Learning and NLP models?

This task is challenging because:

- short texts may contain limited context;
- some emotions can be semantically close;
- the dataset is imbalanced;
- minority classes such as `love` and `surprise` are harder to learn;
- the model must understand the context of words, not only isolated keywords.

---
## Contributors

This project was developed by:

- Ahmed Gaabi
- Mouaffak Yassine Haddar

## Dataset

The dataset used in this project contains short text samples labeled with one emotion.
link dataset : https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?resource=download&select=train.txt
The data is split into three files:

```text
train.txt
val.txt
test.txt
