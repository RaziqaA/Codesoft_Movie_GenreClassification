

---

# Movie Genre Classification

This project builds a machine learning model to classify the genre of a movie based on its description. The dataset used contains movie titles, genres, and descriptions, and the task is to predict the movie's genre using natural language processing (NLP) techniques and classification models.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [How to Use](#how-to-use)
- [Conclusion](#conclusion)

## Overview

This project uses movie descriptions to predict their genre. Different classifiers are used to perform this task, such as:
- Linear Support Vector Classifier (SVC)
- Multinomial Naive Bayes (MNB)
- Logistic Regression (LR)

Natural Language Processing (NLP) techniques such as TF-IDF are used to convert the textual descriptions into a numerical form that can be fed into the classifiers.

## Dataset

The project uses three datasets:
- **train_data.txt**: This dataset contains the movie titles, genres, and descriptions for training.
- **test_data.txt**: This is the test dataset, which includes the movie titles and descriptions. It does not include genres for testing.
- **test_data_solution.txt**: This contains the correct genres for the test dataset, which is used to evaluate model performance.

The data is separated using `:::` as a delimiter.

## Installation

### Prerequisites
- Python 3.x
- Required Libraries:
  ```bash
  pip install pandas numpy seaborn matplotlib scikit-learn
  ```

### Files
Make sure the following files are present in your working directory:
- `train_data.txt`
- `test_data.txt`
- `test_data_solution.txt`

## Data Preprocessing

- **Text Vectorization**: We use `TfidfVectorizer` from `scikit-learn` to transform the movie descriptions into a sparse matrix of TF-IDF features.
- **Label Encoding**: The movie genres are label encoded using `LabelEncoder` to convert text labels into integers for training.
- **Missing Values**: We fill any missing descriptions with an empty string.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Vectorize the text descriptions
t_v = TfidfVectorizer(stop_words='english', max_features=100000)
X_train = t_v.fit_transform(train_data['DESCRIPTION'])
X_test = t_v.transform(test_data['DESCRIPTION'])

# Encode the genres
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['GENRE'])
y_test = label_encoder.transform(test_solution_data['GENRE'])
```

## Exploratory Data Analysis

We performed basic data analysis to explore the distribution of movie genres and the length of movie descriptions. Below are some visualizations:

1. **Distribution of Movies per Genre**:
   ![Movies per Genre]
   
2. **Description Length by Genre**:
   ![Description Length by Genre]

## Modeling

We use different classification models:
1. **LinearSVC**: A linear support vector machine classifier for high-dimensional datasets.
2. **Multinomial Naive Bayes**: A simple probabilistic classifier based on applying Bayes’ theorem.
3. **Logistic Regression**: A widely used linear model for classification.

### Training
```python
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and validation sets
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train LinearSVC model
clf = LinearSVC()
clf.fit(X_train_sub, y_train_sub)
```

### Evaluation

#### Validation Set
```python
# Validate the model
y_val_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))
```

#### Test Set
```python
# Test the model
y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Test Classification Report:\n", classification_report(y_test, y_pred))
```

The best performing model had a validation accuracy of around **58%**. However, the test accuracy was **9%**, which indicates that the model struggled on unseen data.

## How to Use

### Predicting a Movie Genre
You can use the trained model to predict the genre of a new movie description:

```python
def predict_movie(description):
    t_v1 = t_v.transform([description])
    pred_label = clf.predict(t_v1)
    return label_encoder.inverse_transform(pred_label)[0]

sample_description = "A movie where police chase the criminal and shoot him"
print(predict_movie(sample_description))  # Outputs: 'action'
```

### Example Prediction
```python
sample_descr = "A movie where a man falls in love with a woman but faces challenges."
print(predict_movie(sample_descr))  # Outputs: 'drama'
```

## Conclusion

This project demonstrates how movie descriptions can be used to predict their genre using natural language processing and machine learning techniques. While the model performed reasonably well on validation data, the test accuracy suggests further improvement could be made by experimenting with other models, additional features, or more refined text processing techniques.

## Future Improvements
- Experimenting with other classifiers such as Random Forest, Gradient Boosting, or deep learning models.
- More advanced NLP techniques like word embeddings (Word2Vec, GloVe) or transformer models (BERT).
- Tuning hyperparameters of models to improve performance.

---

