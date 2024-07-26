import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
# label encoding
from sklearn.preprocessing import LabelEncoder
# test train split
from sklearn.model_selection import train_test_split
# this is used for dimension reduction to prevent overfitting and mitigate the curse of dimensionality
from sklearn.feature_extraction.text import TfidfVectorizer

# the kaggle dataset has 3 columns, "Unnamed: 0", "Email Text", "Email Type" 

def preprocess_data():
    # load data
    data = pd.read_csv("data/kaggle_phishing_email.csv")
    # drop null values and duplicates
    data = data.drop(columns=["Unnamed: 0"], axis=1)
    data = data.dropna()
    data = data.drop_duplicates()
    # remove links, punctuation, extra spaces, and convert to lowercase
    data["Email Text"] = data["Email Text"].apply(lambda x: re.sub(r'http\S+', '', x))
    data["Email Text"] = data["Email Text"].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    data["Email Text"] = data["Email Text"].apply(lambda x: re.sub(r'\s+', ' ', x))
    data["Email Text"] = data["Email Text"].apply(lambda x: x.strip())
    data["Email Text"] = data["Email Text"].apply(lambda x: x.lower())
    # label encoding: phishing = 0, safe = 1
    le = LabelEncoder()
    data["Email Type"] = le.fit_transform(data["Email Type"])
    return data

def vectorize_data(data):
    # vectorizer = TfidfVectorizer()
    # vectorize data and dimension reduction
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X = vectorizer.fit_transform(data["Email Text"]).toarray()
    y = np.array(data["Email Type"])
    return X, y

def split_data(X, y):
    # split data into train (70%), validation (15%), test sets (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_data():
    data = preprocess_data()
    X, y = vectorize_data(data)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# if __name__ == '__main__':
#     data = preprocess_data()
#     # print how many phishing and safe emails
#     print(data["Email Type"].value_counts())
#     print(data.head())
#     print(data.columns)
#     # display distribution
#     plt.pie(data["Email Type"].value_counts(), labels=["Phishing", "Safe"], autopct='%1.1f%%')
#     plt.title('Categorical Distribution of Safe and Phishing Emails')
#     plt.show()
#     X, y = vectorize_data(data)
#     X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    