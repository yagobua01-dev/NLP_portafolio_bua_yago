import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
# Suppress TensorFlow logging warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Data Loading and Preparation
# Read the CSV files
train_df = pd.read_csv('sent_train.csv')
valid_df = pd.read_csv('sent_valid.csv')