from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import load_data 

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification:')
print(report)
