#Program in python to detect spam messages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string


# Download the stopwords from NLTK
nltk.download('stopwords')

# Load the dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Basic text preprocessing function
def preprocess_text(text):
    # Convert text to lower case
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Apply preprocessing to the messages
df['message'] = df['message'].apply(preprocess_text)

# Split the data into features and labels
X = df['message']
y = df['label']

# Convert text data into numerical data using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Function to classify new messages
def classify_message(message):
    message = preprocess_text(message)
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

# Example usage
example_message = input("Enter your message : ")
print(f'The message "{example_message}" is classified as: {classify_message(example_message)}')


#Example message:As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589


