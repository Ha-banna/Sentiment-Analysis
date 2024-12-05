# Sentiment Analysis on Yelp Reviews

This project performs sentiment analysis on Yelp reviews using **Natural Language Processing (NLP)** techniques. It includes text preprocessing using **tokenization** and **Bag of Words**, and a **Logistic Regression** model to classify reviews as **positive** or **negative**. The project leverages Python libraries such as `nltk`, `spaCy`, and `scikit-learn` for preprocessing, model training, and evaluation.

## Project Overview

This project uses the following steps to classify Yelp reviews:

1. **Text Preprocessing**:
   - Tokenization of text data
   - Removal of stop words and punctuation
   - Lemmatization of words using **spaCy** and **NLTK**
   - Text vectorization using **CountVectorizer** (Bag of Words method)

2. **Model Training**:
   - Logistic Regression model to predict sentiment (positive or negative)

3. **Model Evaluation**:
   - Model performance is evaluated using metrics such as **accuracy**, **precision**, **recall**, and **confusion matrix**.

---

## Requirements

To run the project locally, you will need the following dependencies:

- **Python 3.x**
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `spacy`
  - `wordcloud`
  - `scikit-learn`

## Running the Project

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Ha-banna/Sentiment-Analysis.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Sentiment-Analysis
   ```

3. Install the necessary libraries if you haven't already:


4. Run the Jupyter notebook for training the model:

   ```bash
   jupyter notebook Sentiment_Analysis_(Final).ipynb
   ```

---

## Project Files

- **Sentiment_Analysis_(Final).ipynb**: Jupyter notebook with the implementation for text preprocessing, model training, and evaluation.

---

## Project Steps

1. **Data Preprocessing**: 
   - The dataset is preprocessed to remove stop words, tokenize the text, and lemmatize words. We use **NLTK** and **spaCy** for text cleaning and lemmatization.

2. **Vectorization**: 
   - The text data is converted into numerical format using **CountVectorizer**, which implements the **Bag of Words** model.

3. **Training the Model**: 
   - The **Logistic Regression** classifier is trained on the preprocessed and vectorized data.

4. **Model Evaluation**: 
   - The model's performance is evaluated using **accuracy**, **precision**, **recall**, and a **confusion matrix**.

---

## License

This project is open-source and available under the MIT License.
