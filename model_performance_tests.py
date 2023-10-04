# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:22:11 2023

Author: Grace Milner

This script allows the user to run the classification model multiple times, going through each 
input file in turn (there are separate files for the different relevance types of tweets)
with different model configurations. The only thing that needs to be determined by the user 
before running is the type of preproccessing required (adjust the custom tokenizer function 
accordingly). 

Performance metrics are calculated and saved in a new dataframe. There is also the option to
export the results to excel. This allows the comparison of different model configurations
to determine the most appropriate in different contexts (for tweets of different R types)

"""

import os
import pandas as pd 
import spacy
import spacy.lang.en.stop_words
import string 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
#import xlsxwriter

# Loading SpaCy's English NLP model (small version, no word vectors)
nlp = spacy.load('en_core_web_sm')
# Create list of punctuation marks
punctuations = string.punctuation
# Adding TM to punctuation (emerged as an issue during testing)
punctuations += "™"

# Create list of stop words
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Create custom tokenizer function
def spacy_tokenizer(sentence):
    # Creating token object, used to create documents with linguistic annotations (gives insights into grammatical structure).
    # Disabling unneeded parts of the pipeline for now (parser, and entity recognition)
    mytokens = nlp(sentence, disable=['parser', 'ner'])
    
    # Lemmatising each token if required
    # (spaCy uses '-PRON-' to refer to all personal pronouns e.g. 'I, me')
    # Converts token to lemmatised version, if pronoun not lemmatised. Also makes lowercase and removes white spaces. 
    mytokens = [ word.lemma_.lower().strip() for word in mytokens if word.lemma_ != "-PRON-" ] 
    # Removing stop words and punctuation 
    # (this removes any token which contains any punctuation, not just the punctuation, so also removes usernames or links.)    
    mytokens = [ word for word in mytokens if word not in stop_words and not any([x in word for x in punctuations])]
    # return preprocessed list of tokens
    return mytokens

# Creating a function to fit and evaluate logistic regression and SVM models
def evaluate_model(file, X_train, y_train, X_test, y_test, vector_type, ngram_type, model_type):
    # Setting model type
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'svm':
        model = SVC(kernel='rbf')
    else:
        print('Model type not recognized.')
        return None
    # Setting ngram type
    if ngram_type == 'unigram':
        ngram = (1,1)
    elif ngram_type == 'uni + bigram':
        ngram = (1,2)
    else:
         print('Ngram type not recognized.')
         return None
    # Setting vectoriser type
    if vector_type =='bow_vectors':
        vectors = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range = ngram)
    elif vector_type == 'tfidf_vectors':
        vectors = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range = ngram)
    else:
         print('Vector type not recognized.')
         return None
    # Defining pipeline 
    pipe = Pipeline([
        ('vectorizer', vectors),
        ('model', model)
    ])
    # Running pipeline with training data
    pipe.fit(X_train, y_train)
    # Running pipeline with testing data
    y_pred = pipe.predict(X_test)
    
    results = {
        'data': str(file)[:-4],
        'model_type': model_type,
        'vector_type': vector_type,
        'ngram_type': ngram_type,
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred),
        'recall': metrics.recall_score(y_test, y_pred),
        'f1_score': metrics.f1_score(y_test, y_pred)
    }
    
    return results


# Create empty results DataFrame to store results
results_df = pd.DataFrame(columns=['data', 'model_type', 'vector_type', 'ngram_type', 'accuracy', 'precision', 'recall', 'f1_score'])

# Set working directory to the directory containing the CSV files
os.chdir(r"C:\VUB\Thesis\Data\Tweets\For model")


# Loop through CSV files in the directory
for file in os.listdir():
    if file.endswith('.csv'):
        print(f'Processing file {file}...')
        
        # Read in the CSV file
        df = pd.read_csv(file)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['Tweet'], df['Relevance'], test_size=0.3, random_state=1)
        
        # Fit and evaluate logistic regression and SVM models with different vectorization methods and ngram ranges
        for ngram_type in ['unigram', 'uni + bigram']:
            for vector_type in ['bow_vectors', 'tfidf_vectors']:
                for model_type in ['logistic_regression', 'svm']:
                    # Fit and evaluate the model
                    results = evaluate_model(file, X_train, y_train, X_test, y_test, vector_type, ngram_type, model_type)
                    # Add the results to the results
                    results_df = results_df.append(results, ignore_index=True)
                    
#results_df.to_excel('full_preprocessing_results.xlsx', index=False)
