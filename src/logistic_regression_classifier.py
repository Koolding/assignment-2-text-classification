# import packages
import os

# data munging tools
import pandas as pd

# machine learning tools
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# save model and report
from joblib import dump, load

# for scripting
import argparse


# parser function
def input_parse():
    parser = argparse.ArgumentParser()
    #add arguments
    parser.add_argument("--filename", type=str, default= "fake_or_real_news.csv", help= "Specify .csv file.") 
    parser.add_argument("--test_size", type= float, default =0.2, help= "Specify the proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type= int, default = 666, help = "Specify random state.")
    parser.add_argument("--ngram_range", nargs='+', type= int, default=(1,2), help = "Specify the lower and upper boundary of the range of n-values for different n-grams to be extracted.")
    parser.add_argument("--lowercase", type= bool, default = True, help = "Specify whether lowercase should be True or False.")
    parser.add_argument("--max_df", type= float, default = 0.95, help= "Specify maximum threshold and ignore terms that have a document frequency strictly higher than the given threshold.")
    parser.add_argument("--min_df", type= float, default = 0.05, help = "Specify minimum threshold and ignore terms that have a document frequency strictly lower than the given threshold.")
    parser.add_argument("--max_features", type= int, default = 500, help = "Specify maximum number of features to extract.")
    
    
    # parse arguments from the command line 
    args = parser.parse_args()
    
    # return value
    return args



# load in data
import vectorizer as vec
args = input_parse()
X,y = vec.load_data(args.filename)
X_train_feats, X_test_feats, y_train, y_test, vectorizer = vec.vectorize_function(X, y, args.test_size, args.random_state, args.ngram_range, args.lowercase, args.max_df, args.min_df, args.max_features)
 

# classify and predict
def log_reg_model(random_state):

    print("Initializing logistic regression classifier..")

    classifier = LogisticRegression(random_state = random_state).fit(X_train_feats, y_train) 
    

    # predictions of y 
    y_pred = classifier.predict(X_test_feats) 



    # evaluate
    # Calculating metrics for model performance
    classifier_metrics = metrics.classification_report(y_test, y_pred) 

    return(classifier_metrics, classifier)




# save model and metrics report
def save_LR_results(classifier_metrics, classifier):

    outpath_metrics_report = os.path.join(os.getcwd(), "out", "LR_metrics_report.txt")

    file = open(outpath_metrics_report, "w")
    file.write(classifier_metrics)
    file.close()

    outpath_classifier = os.path.join(os.getcwd(), "models", "LR_classifier.joblib")

    dump(classifier, open(outpath_classifier, 'wb'))

    print( "The logistic regression metrics report is saved in the folder ´out´")
    print( "The logistic regression model is saved in the folder ´models´")

    

# main function
def main():
    args = input_parse()
    classification_metrics, classifier = log_reg_model(args.random_state)
    save_LR_results(classification_metrics, classifier)

if __name__ == '__main__':
    main()
