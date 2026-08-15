import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# 1. LOAD SMS SPAM DATASET
# =========================================================

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("========== First 5 Rows ==========")
print(df.head())


# =========================================================
# 2. DATASET INFORMATION
# =========================================================

print("\n========== Dataset Info ==========")
df.info()


# =========================================================
# 3. CHECK MISSING VALUES
# =========================================================

print("\n========== Missing Values ==========")
print(df.isnull().sum())


# =========================================================
# 4. SPAM / HAM COUNT
# =========================================================

print("\n========== Spam / Ham Count ==========")
print(df["label"].value_counts())


# =========================================================
# 5. DATASET SHAPE
# =========================================================

print("\n========== Dataset Shape ==========")
print(df.shape)


# =========================================================
# 6. TEXT PREPROCESSING
# =========================================================

def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Apply cleaning function
df["clean_message"] = df["message"].apply(clean_text)


# Display original and cleaned messages
print("\n========== Original vs Cleaned Messages ==========")

for i in range(5):
    print("Original :", df["message"].iloc[i])
    print("Cleaned  :", df["clean_message"].iloc[i])
    print()


# =========================================================
# 7. TF-IDF VECTORIZATION
# =========================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_message"])

print("\n========== TF-IDF ==========")
print("Number of messages:", X.shape[0])
print("Number of features:", X.shape[1])


# =========================================================
# 8. LABELS
# =========================================================

y = df["label"]


# =========================================================
# 9. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== Train-Test Split ==========")
print("Training messages:", X_train.shape[0])
print("Testing messages :", X_test.shape[0])


# =========================================================
# 10. NAIVE BAYES MODEL
# =========================================================

model = MultinomialNB()

model.fit(X_train, y_train)

print("\n========== Model Training ==========")
print("Naive Bayes model trained successfully!")


# =========================================================
# 11. PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

print("\n========== Predictions ==========")

for actual, predicted in zip(y_test.iloc[:10], y_pred[:10]):
    print("Actual:", actual, "| Predicted:", predicted)


# =========================================================
# 12. MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== Evaluation ==========")
print("Accuracy:", accuracy)


print("\n========== Classification Report ==========")
print(classification_report(y_test, y_pred))


print("\n========== Confusion Matrix ==========")
print(confusion_matrix(y_test, y_pred))


# =========================================================
# 13. TEST WITH OUR OWN SMS
# =========================================================

test_messages = [
    "Congratulations! You won a free prize. Call now!",
    "Hey, are you coming to college today?",
    "You have won 1000 dollars. Click the link to claim.",
    "Can you call me when you reach home?"
]

# Clean test messages
clean_test_messages = [clean_text(message) for message in test_messages]

# Convert messages into TF-IDF numbers
test_vectors = vectorizer.transform(clean_test_messages)

# Predict
predictions = model.predict(test_vectors)

print("\n========== Custom Message Predictions ==========")

for message, prediction in zip(test_messages, predictions):
    print("Message:", message)
    print("Prediction:", prediction.upper())
    print()