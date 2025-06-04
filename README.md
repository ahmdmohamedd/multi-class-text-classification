# BERT-Based News Classification

This project implements a BERT-based model to classify news articles into multiple categories using their titles and descriptions. Leveraging Hugging Face's `transformers` and `datasets` libraries, it demonstrates an end-to-end NLP pipeline, from preprocessing raw CSV data to fine-tuning a pre-trained language model.

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Training Configuration](#training-configuration)
* [Results](#results)
* [Limitations](#limitations)


## Project Overview

This project focuses on multi-class classification of news articles using a BERT model (`bert-base-uncased`). The pipeline includes:

* Data loading and preprocessing with `pandas`
* Dataset conversion using Hugging Face's `datasets` library
* Tokenization with `BertTokenizer`
* Model fine-tuning with `Trainer`
* Evaluation setup for accuracy and F1 score

## Dataset

The dataset consists of two CSV files:

* `train.csv`
* `test.csv`

Each file contains:

* `Class Index`: Label for the article (integer)
* `Title`: The news title
* `Description`: A short summary of the article

The `Title` and `Description` are concatenated into a single text input for the model.

## Installation

Create a new environment and install the required packages:

```bash
conda create -n news-classifier python=3.9
conda activate news-classifier
pip install transformers datasets pandas scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/ahmdmohamedd/multi-class-text-classification.git
cd multi-class-text-classification
```

2. Place the `train.csv` and `test.csv` files in the project directory.

3. Launch Jupyter Notebook and run the `.ipynb` file:

```bash
jupyter notebook bert_news_classifier.ipynb
```

The notebook will guide you through:

* Data preprocessing
* Dataset tokenization
* Model training
* Evaluation metrics

## Model Architecture

The model is based on `bert-base-uncased` with a classification head added on top. It uses `BertForSequenceClassification` from the `transformers` library, configured for multi-class classification.

## Training Configuration

Key training arguments include:

* Epochs: 3
* Batch size: 8
* Learning rate: 5e-5
* Evaluation strategy: Per epoch
* Metric: Accuracy and F1

Training was conducted using `Trainer`, allowing easy integration of logging, evaluation, and checkpointing.

## Results

The training pipeline is functional and evaluation-ready. Due to hardware limitations, the full training process was not completed, but all components are validated and working correctly. You can continue training with more resources or integrate early stopping/checkpoint resumption.

## Limitations

* Training time is significant on non-GPU or limited-resource environments
* Dataset assumes clean, labeled input
* The model may overfit if trained too long on a small dataset

---
