# Week 3 - SMS Spam Classifier

## Objective

Build a machine learning model to classify SMS messages as **Spam** or **Ham** using Text Preprocessing, TF-IDF and Naive Bayes.

## Dataset

SMS Spam Collection Dataset

- Total Messages: 5572
- Ham Messages: 4825
- Spam Messages: 747

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes

## Steps

1. Load the SMS dataset
2. Check and analyze the data
3. Clean and preprocess SMS text
4. Convert text into numerical features using TF-IDF
5. Split data into training and testing sets
6. Train the Naive Bayes model
7. Predict Spam/Ham messages
8. Evaluate model performance

## Train-Test Split

- Training Messages: 4457
- Testing Messages: 1115

## Results

**Accuracy: 95.78%**

### Confusion Matrix

```text
[[966   0]
 [ 47 102]]
