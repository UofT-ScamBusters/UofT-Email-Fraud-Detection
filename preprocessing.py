import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from wordcloud import WordCloud
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

    ###############################################################

    # load uoft data
    uoft_data = pd.read_csv("data/uoft_phishing_email.csv")
    uoft_data = uoft_data.drop(columns=["Unnamed: 0"], axis=1)
    uoft_data = uoft_data.dropna()
    uoft_data = uoft_data.drop_duplicates()
    uoft_data["Email Text"] = uoft_data["Email Text"].apply(lambda x: re.sub(r'http\S+', '', x))
    uoft_data["Email Text"] = uoft_data["Email Text"].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    uoft_data["Email Text"] = uoft_data["Email Text"].apply(lambda x: re.sub(r'\s+', ' ', x))
    uoft_data["Email Text"] = uoft_data["Email Text"].apply(lambda x: x.strip())
    uoft_data["Email Text"] = uoft_data["Email Text"].apply(lambda x: x.lower())
    uoft_data["Email Type"] = le.fit_transform(uoft_data["Email Type"])

    return data, uoft_data

def vectorize_data(data, uoft_data=None):
    # vectorizer = TfidfVectorizer()
    # vectorize data and dimension reduction
    # ignore common English stopwords when transforming the text data
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X_train = vectorizer.fit_transform(data["Email Text"]).toarray()
    y_train = np.array(data["Email Type"])
    
    if uoft_data is not None:
        X_uoft = vectorizer.transform(uoft_data["Email Text"]).toarray()
        y_uoft = np.array(uoft_data["Email Type"])
        return X_train, X_uoft, y_train, y_uoft
    
    return X_train, y_train

def split_data(X, y):
    # split data into train (70%), validation (15%), test sets (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_data():
    kaggle_data, uoft_data = preprocess_data()
    X, X_uoft, y, y_uoft = vectorize_data(kaggle_data, uoft_data)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_data_uoft_kaggle_merged_test():
    kaggle_data, uoft_data = preprocess_data()
    X, X_uoft, y, y_uoft = vectorize_data(kaggle_data, uoft_data)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    # include uoft data in test set
    X_test = np.concatenate((X_test, X_uoft), axis=0)
    y_test = np.concatenate((y_test, y_uoft), axis=0)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def load_data_uoft_kaggle_separate_test():
    kaggle_data, uoft_data = preprocess_data()
    X, X_uoft, y, y_uoft = vectorize_data(kaggle_data, uoft_data)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    # have 2 test sets, one for kaggle and one for uoft
    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_uoft, y_uoft

def create_wordcloud(data, title, filter_label=None, file_name=None):
    if filter_label is not None:
        filtered_data = data[data["Email Type"] == filter_label]
    else:
        filtered_data = data
    
    text = " ".join(filtered_data["Email Text"])
    wordcloud = WordCloud(width=800, height=800, 
                          background_color='white', 
                          stopwords=None, 
                          min_font_size=10).generate(text)
    
    plt.figure(figsize=(8, 8), facecolor=None) 
    plt.imshow(wordcloud, interpolation="bilinear") 
    plt.axis("off")
    plt.title(title, fontsize=20)
    plt.tight_layout(pad=0) 
    if file_name:
        plt.savefig(file_name)
    plt.show()

def show_distribution(data, title, file_name=None):
    plt.pie(data["Email Type"].value_counts(), labels=["Phishing", "Safe"], autopct='%1.1f%%')
    plt.title(title)
    if file_name:
        plt.savefig(file_name)
    plt.show()

def show_visualizations():
    kaggle_data, uoft_data = preprocess_data()

    # Create directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)

    # distribution of phishing and safe emails
    show_distribution(kaggle_data, "Kaggle Dataset Categorical Distribution of Safe and Phishing Emails", file_name="visualizations/kaggle_distribution.png")
    show_distribution(uoft_data, "UofT Dataset Categorical Distribution of Safe and Phishing Emails", file_name="visualizations/uoft_distribution.png")

    # word clouds for phishing emails
    create_wordcloud(kaggle_data, "Kaggle Dataset Phishing Email WordCloud", filter_label=0, file_name="visualizations/kaggle_phishing_wordcloud.png")
    create_wordcloud(uoft_data, "UofT Dataset Phishing Email WordCloud", filter_label=0, file_name="visualizations/uoft_phishing_wordcloud.png")

    # word clouds for safe emails
    create_wordcloud(kaggle_data, "Kaggle Dataset Safe Email WordCloud", filter_label=1, file_name="visualizations/kaggle_safe_wordcloud.png")
    create_wordcloud(uoft_data, "UofT Dataset Safe Email WordCloud", filter_label=1, file_name="visualizations/uoft_safe_wordcloud.png")

if __name__ == '__main__':
    show_visualizations()
    
    