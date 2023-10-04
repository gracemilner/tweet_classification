# tweet_classification
Machine Learning model used to classify tweets into relevant or irrelevant based on text content.

## About
Machine learning model to classify tweets into relevant (1) or irrelevant (0) based on text content. This project formed part of a final Thesis project regarding the identification of urban gully events in Nigeria through social media.

The following model configurations can be manually changed by the user and then the model run through once:
    (1) Classifier -> Logistic Regression or SVM
    (2) Pre-processing steps (none, no lemmatisation, or full)
    (3) n-gram range (unigram or unigram + bigram)
    (4) Vectoriser (BoW or BoW with TFIDF weighting)

Final section provides option to test to see the prediction success per R (relevance) type when all R types are classified together. 
The optimal configuration for each R type can be chosen and the results compared. 
    E.g. can be used to compare the recall of R1 tweets when just classified individually, or when classified 
    within the R123 (all relevance kinds combined). 

Additional script available to conduct model performance tests and compare different model configurations. The performances are assessed with a range of metrics to determine the most appropriate in different contexts.
