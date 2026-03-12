# Sentiment Analysis using BiLSTM

This project implements a deep learning based Natural Language Processing (NLP) pipeline for movie review sentiment classification using a Bidirectional LSTM (BiLSTM) model built with TensorFlow and Keras.

The model is trained on the IMDB Movie Review Dataset to classify reviews as Positive or Negative.

--------------------------------------------------

## Project Highlights

- End-to-end NLP pipeline
- Text preprocessing and cleaning
- Tokenization and sequence padding
- Word embeddings using neural networks
- Bidirectional LSTM for contextual learning
- Model training and inference
- Modular project structure

--------------------------------------------------

## Problem Statement

Understanding sentiment in textual data is an important task in modern applications such as:

- Product review analysis
- Social media sentiment monitoring
- Customer feedback analysis
- Market research

This project builds a deep learning model that can automatically determine whether a review expresses positive or negative sentiment.

--------------------------------------------------

## Dataset

Dataset Used: IMDB Movie Review Dataset

Dataset details:

Total Reviews: 50,000  
Training Samples: 25,000  
Testing Samples: 25,000  
Classes: Positive / Negative

Example Data:

Review: "This movie was amazing"  
Sentiment: Positive

Review: "Worst movie ever"  
Sentiment: Negative

--------------------------------------------------

## System Architecture

NLP Pipeline:

Raw Text  
↓  
Text Cleaning  
↓  
Tokenization  
↓  
Word Index Encoding  
↓  
Sequence Padding  
↓  
Embedding Layer  
↓  
Bidirectional LSTM  
↓  
Dense Layer  
↓  
Sigmoid Output  
↓  
Sentiment Prediction

--------------------------------------------------

## Model Architecture

Embedding Layer  
Input Dimension: Vocabulary Size  
Output Dimension: 128

Bidirectional LSTM  
Units: 64

Dense Layer  
Activation: ReLU

Output Layer  
Activation: Sigmoid

Loss Function: Binary Cross Entropy  
Optimizer: Adam

--------------------------------------------------

## Project Structure

sentiment-analysis-bilstm

data  
 └── IMDB_Dataset.csv  

models  
 ├── sentiment_model.h5  
 └── tokenizer.pkl  

src  
 ├── data_preprocessing.py  
 ├── train_model.py  
 └── predict.py  

requirements.txt  
README.md

--------------------------------------------------

## Installation

Clone the repository

git clone https://github.com/AravindKKrishna/sentiment-analysis-bilstm.git

Navigate to project directory

cd sentiment-analysis-bilstm

Install dependencies

pip install -r requirements.txt

--------------------------------------------------

## Training the Model

Run the training script

cd src  
python train_model.py

This will perform:

- Data preprocessing
- Tokenization
- Model training
- Model saving

Generated files:

models/sentiment_model.h5  
models/tokenizer.pkl

--------------------------------------------------

## Running Predictions

Run prediction script

python predict.py

Example input:

This movie was fantastic and exciting

Output:

Positive

--------------------------------------------------

## Model Performance

Expected Performance

Accuracy: ~85% – 90%  
Loss Function: Binary Cross Entropy

--------------------------------------------------

## Key Concepts Used

- Text preprocessing
- Stopword removal
- Tokenization
- Word embeddings
- Sequence modeling
- Deep learning for text classification

--------------------------------------------------

## Future Improvements

Possible improvements:

- Add Attention Layer
- Use pretrained embeddings like GloVe
- Deploy model using Flask API
- Build a web interface
- Convert model to REST API
- Use Transformer based architectures

--------------------------------------------------

## Applications

This project can be applied to:

- Product review analysis
- Social media sentiment detection
- Customer feedback classification
- Brand monitoring

--------------------------------------------------

## Author

Developed as an NLP deep learning project using TensorFlow and LSTM architectures.
