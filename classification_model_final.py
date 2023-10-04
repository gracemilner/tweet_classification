# -*- coding: utf-8 -*-
"""

Author: Grace Milner

Machine learning model to classify tweets into relevant (1) or irrelevant (0) based on text content. 

The following model configurations can be manually changed by the user and then the model run through once:
    (1) Classifier -> Logistic Regression or SVM
    (2) Pre-processing steps (none, no lemmatisation, or full)
    (3) n-gram range (unigram or unigram + bigram)
    (4) Vectoriser (BoW or BoW with TFIDF weighting)

Final section provides option to test to see the prediction success per R type when all R types are classified together. 
The optimal configuration for each R type can be chosen and the results compared. 
    E.g. can be used to compare the recall of R1 tweets when just classified individually, or when classified 
    within the R123 (all relevance kinds combined). 

"""


# Loading in libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics


# =============================================================================
#           INITIAL TEXT PROCESSING 
# =============================================================================

# Reading in the data
os.chdir("C:\VUB\Thesis\Data\Tweets\For model")
df = pd.read_csv("R123_new.csv")

# Look at top 5 records
print(df.head())
# Look at shape of df
print(df.shape)
# Can also value count to see how many in this dataset are classed as relevant
print(df["Relevance"].value_counts())

# Import required libraries
import spacy
import string 
    #includes list of punctuation marks
import spacy.lang.en.stop_words
    #model to remove common stop words 

# Create list of punctuation marks
punctuations = string.punctuation
# Adding TM to punctuation (emerged as an issue during testing)
punctuations += "™"

# Create list of stop words
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Loading SpaCy's English NLP model (small version, no word vectors)
nlp = spacy.load('en_core_web_sm')

# Create custom tokeniser function:
#       Will take a sentence (or tweet) as input and process
#       into tokens, remove stop words, turn into 
#       lowercase and perform lemmatisation.


def spacy_tokenizer(sentence):
    # Creating token object, used to create documents with linguistic annotations (gives insights into grammatical structure).
    # Disabling unneeded parts of the pipeline for now (parser, and entity recognition)
    mytokens = nlp(sentence, disable=['parser', 'ner'])

    # Lemmatising each token if required (uncomment when needed)
    # (spaCy uses '-PRON-' to refer to all personal pronouns e.g. 'I, me')
    # Converts token to lemmatised version, if pronoun not lemmatised. Also makes lowercase and removes white spaces. 
    #mytokens = [ word.lemma_.lower().strip() for word in mytokens if word.lemma_ != "-PRON-"]

    # Removing stop words and punctuation 
    # (this removes any token which contains any punctuation, not just the punctuation, so also removes usernames or links.)    
    #mytokens = [ word for word in mytokens if word not in stop_words and not any([x in word for x in punctuations])]
    
    # If removing pre-processing steps, need to manually convert tokens to strings:
    mytokens = [str(token) for token in mytokens]
    
    # return preprocessed list of tokens
    return mytokens


##### Vectorising Data #####
# Converting data into machine-readable format. 

# Using Bag of Words (BoW) to convert text to numeric (vector) format
    #(generates matrix which records occurrence of words within document)

# Generating vector by using CountVectorizer from scikit-learn
#   Uses our custom tokenisor, and defines the ngram range (combination of adjacent words)
#   Creates a BoW matrix where each unique word given a column and each text sample (tweet) given a row.
#   Populated by 0 or 1 depending on if the tweet contains that specific word.  


bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,1))
# Used unigram here, so lower and upper bound of ngram range = 1. Could change to (2, 2) to only use bigrams.
#   Important to note, ngram range impacts the corpus not the vocabulary. 

# Using Term Frequency-Inverse Document Frequency (TF-IDF) to normalise BoW output. 
#   Higher TF-IDF value --> term more important to the document
# If using this normalised version, need to replace it for bow_vector in pipeline
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,2)) #again can change ngram range.


# =============================================================================
#           TRAIN AND TEST DATASETS
# =============================================================================

# Generating training data to train model and test dataset to test model performance

from sklearn.model_selection import train_test_split

X = df['Tweet'] # The features we want to analyse
ylabels = df['Relevance'] # The labels/target variables, relevance

# Splitting up data using sklearn
#   Takes input features, labels, and test size as arguments.
#   Test size gives percentage of split e.g. 0.3 -> 70% training data and 30% test data
#   Random state argument optional, used for initialising internal random number gnerator
#       which decides splitting of data. random_state=1 means each run of code will give
#       same data in training and test split.
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size = 0.3, random_state = 1)


# =============================================================================
#           GENERATING MODEL - LOGISTIC REGRESSION
# =============================================================================

classifier_lr = LogisticRegression()
pipe_lr = Pipeline([
    ("vectorizer", tfidf_vector),
    ("classifier", classifier_lr)
])
pipe_lr.fit(X_train, y_train)

# Predicting with a test dataset (named lr to show log reg)
predicted_lr = pipe_lr.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted_lr))
# Model Precision
print("Logistic Regression Precision:", metrics.precision_score(y_test, predicted_lr))
# Model Recall
print("Logistic Regression Recall:", metrics.recall_score(y_test, predicted_lr))
# F1 Score
print("Logistic Regression F1:", metrics.f1_score(y_test, predicted_lr))


# =============================================================================
#           GENERATING MODEL - SVM
# =============================================================================

classifier_svm = svm.SVC(kernel='rbf')
pipe_svm = Pipeline([
    ("vectorizer", tfidf_vector),
    ("classifier", classifier_svm)
])
pipe_svm.fit(X_train, y_train)

# Predicting the response for test dataset
predicted_svm = pipe_svm.predict(X_test)

# Model Accuracy
print("SVM Accuracy:", metrics.accuracy_score(y_test, predicted_svm))
# Model Precision
print("SVM Precision:", metrics.precision_score(y_test, predicted_svm))
# Model Recall
print("SVM Recall:", metrics.recall_score(y_test, predicted_svm))
# F1 Score
print("SVM F1:", metrics.f1_score(y_test, predicted_svm))


# =============================================================================
#           OUTPUT DATAFRAME
# =============================================================================

# Creating a new dataframe for predictions
output_df = pd.DataFrame(X_test)
output_df["Actually Relevant"] = y_test
output_df["R1"] = df.loc[X_test.index, "R1"] # used the loc function to match the indices in the input dataset
output_df["R2"] = df.loc[X_test.index, "R2"]
output_df["R3"] = df.loc[X_test.index, "R3"]
output_df["LR Predictions"] = predicted_lr
output_df["LR Correct"] = output_df["Actually Relevant"] == output_df["LR Predictions"]
output_df["SVM Predictions"] = predicted_svm
output_df["SVM Correct"] = output_df["Actually Relevant"] == output_df["SVM Predictions"]

print(output_df.head())



# =============================================================================
#          CALCULATING METRICS (for chosen configuration)
# =============================================================================

# Can use this section of the code to test to see the prediction success per R type when all R types are classified together.
#   Example here is testing R1 type tweets with the SVM model run as configured above. 

# Grouping the output dataframe by R1 and calculating metrics (focussing on SVM results for above configuration)

filtered_df = output_df[output_df["R1"] == 1]

# Calculate metrics using filtered data
accuracy = metrics.accuracy_score(filtered_df["Actually Relevant"], filtered_df["SVM Predictions"])
precision = metrics.precision_score(filtered_df["Actually Relevant"], filtered_df["SVM Predictions"])
recall = metrics.recall_score(filtered_df["Actually Relevant"], filtered_df["SVM Predictions"])
f1_score = metrics.f1_score(filtered_df["Actually Relevant"], filtered_df["SVM Predictions"])

print("Metrics for R1:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)








