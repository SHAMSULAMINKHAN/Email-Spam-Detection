# Email-Spam-Detection
Spam email classifier built with Python and Scikit-learn, using CountVectorizer and Multinomial Naive Bayes to achieve high accuracy on text data.


📧 Spam Email Classifier

A simple and effective machine learning project that classifies emails as spam or not spam (ham) using Natural Language Processing (NLP) techniques and a Naive Bayes model.

🚀 Overview

This project uses a dataset of labeled emails to train a model that can automatically detect spam messages. It leverages text vectorization and probabilistic classification to achieve high accuracy.

🧠 Features
    📩 Classifies emails into Spam or Not Spam
    ⚡ Fast and lightweight model using Multinomial Naive Bayes
    🔤 Text preprocessing with CountVectorizer
    📊 Achieves ~97.7% accuracy on test data
    🧪 Easy to test with custom email inputs
🛠️ Tech Stack
    Python 3
    NumPy
    Pandas
    Scikit-learn
    
📂 Dataset
The dataset used is a CSV file containing:
Category → Label (Spam / Not Spam)
Message → Email text

Example:
Category, Message
ham, "Hey, are we still meeting today?"
spam, "Win a free ticket now!!!"

Install dependencies:
pip install numpy pandas scikit-learn

▶️ Usage
Run the script or notebook to train the model:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('spam.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['Message'], data['Category'], test_size=0.25
)

# Build pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Evaluate
print("Accuracy:", model.score(X_test, y_test))
🧪 Example Prediction
emails = [
    "Sounds great! Are you home now?",
    "Win money now!!! Click here to claim your prize"
]

predictions = model.predict(emails)
print(predictions)

Output:

['ham', 'spam']
📊 Results
Model: Multinomial Naive Bayes
Accuracy: ~97.7%
📁 Project Structure
├── spam.csv
├── spam_classifier.ipynb
├── README.md
🤝 Contributing

Contributions are welcome! Feel free to:

Open an issue
Submit a pull request
Suggest improvements
