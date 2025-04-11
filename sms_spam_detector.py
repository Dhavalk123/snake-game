import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load sample dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", header=None, names=['label', 'message'])

# Convert labels to binary
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
predictions = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, predictions))

# Test manually
while True:
    user_input = input("\nEnter a message to check (or 'exit'): ")
    if user_input.lower() == 'exit':
        break
    user_vec = vectorizer.transform([user_input])
    result = model.predict(user_vec)[0]
    print("Spam" if result == 1 else "Not Spam")
