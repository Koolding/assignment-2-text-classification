# import packages
import os

# data munging tools
import pandas as pd

# saving model and report
from joblib import dump, load

# machine learning tools
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# for scripting
import argparse


# Parser function
def input_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default= "fake_or_real_news.csv", help= "Specify .csv file.") 
    parser.add_argument("--test_size", type= float, default =0.2, help= "Specify the proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type= int, default = 666, help = "Specify random state.")
    parser.add_argument("--ngram_range", nargs='+', type= int, default=(1,2), help = "Specify the lower and upper boundary of the range of n-values for different n-grams to be extracted.")
    parser.add_argument("--lowercase", type= bool, default = True, help = "Specify whether lowercase should be True or False.")
    parser.add_argument("--max_df", type= float, default = 0.95, help= "Specify maximum threshold and ignore terms that have a document frequency strictly higher than the given threshold.")
    parser.add_argument("--min_df", type= float, default = 0.05, help = "Specify minimum threshold and ignore terms that have a document frequency strictly lower than the given threshold.")
    parser.add_argument("--max_features", type= int, default = 500, help = "Specify maximum number of features to extract.")
    parser.add_argument("--activation", type= str, default= "logistic", help= "Speficy activation function for hidden layers.")
    parser.add_argument("--hidden_layer_sizes", nargs='+', type= int, default=(30,30), help="Specify hidden layer sizes. More hidden layers will increase computational time. This must be specified without commas.")
    parser.add_argument("--max_iter", type= int, default=1000, help = "Specify maximum number of iterations.")
    # parse the arguments from the command line 
    args = parser.parse_args()
    
    return args


# load data
import vectorizer as vec
args = input_parse()
X,y = vec.load_data(args.filename)
X_train_feats, X_test_feats, y_train, y_test, vectorizer = vec.vectorize_function(X, y, args.test_size, args.random_state, args.ngram_range, args.lowercase, args.max_df, args.min_df, args.max_features)
 



# classify and predict
def neural_network_model(activation, hidden_layer_sizes, max_iter, random_state):

    print("Initializing neural network classifier..")

    # classify
    classifier = MLPClassifier(activation = activation, # default "logistic"
                            hidden_layer_sizes = hidden_layer_sizes, # default 20 nodes in first hidden layer, 10 in next hidden layer
                            max_iter= max_iter,  # default 1000
                            random_state = random_state) # default 666

    print("Activation =", activation)
    print("Number of hidden layer sizes =", hidden_layer_sizes)
    print("Maximum number of iterations =", str(max_iter))

    classifier.fit(X_train_feats, y_train)

    # predictions of y
    y_pred = classifier.predict(X_test_feats) 


    # evaluate
    classification_metrics = metrics.classification_report(y_test, y_pred) 

    return classification_metrics, classifier



# save report and model
def save_NN_results(classifier_metrics, classifier):

    outpath_metrics_report = os.path.join(os.getcwd(), "out", "NN_metrics_report.txt")

    file = open(outpath_metrics_report, "w")
    file.write(classifier_metrics)
    file.close()

    outpath_classifier = os.path.join(os.getcwd(), "models", "NN_classifier.joblib")

    dump(classifier, open(outpath_classifier, 'wb'))

    print( "The neural network metrics report is saved in the folder ´out´")
    print( "The neural network model is saved in the folder ´models´")



# main function
def main():
    args = input_parse()
    classification_metrics, classifier = neural_network_model(args.activation, tuple(args.hidden_layer_sizes), args.max_iter, args.random_state)
    save_NN_results(classification_metrics, classifier)

if __name__ == '__main__':
    main()



