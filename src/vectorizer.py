# importing packages
import os

# data munging tools
import pandas as pd

# machine learning tools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, ShuffleSplit

# for saving vectorizer
from joblib import dump, load

# for scripting
import argparse


# extraction for reading in data and dividing it into text and label
def load_data(filename):
    
    file = os.path.join("in", filename)
    data = pd.read_csv(file, index_col=0)
    print("Reading in " + filename + "...")

    X = data["text"]
    y = data["label"]

    return X,y 



def vectorize_function(X, y, test_size, random_state, ngram_range, lowercase, max_df, min_df, max_features):

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,          
                                                        y,          
                                                        test_size = test_size,   # default is 80/20 split
                                                        random_state=random_state) # random state for reproducibility

    print("Splitting data with test size " + str(test_size))
    

    # vectorize and feature extraction
    vectorizer = TfidfVectorizer(ngram_range = ngram_range,     
                                lowercase =  lowercase,       
                                max_df = max_df,          
                                min_df = min_df,           
                                max_features = max_features)      # default keeps top 500 features
The scale of the lower and upper range of n-values for different n-grams
    print("The scale of the lower and upper range of n-values for different n-grams = " + str(ngram_range))
    print("Converting the data to lowercase = " + str(lowercase))
    print("Setting a max_df threshold = " + str(max_df))
    print("Setting a min_df threshold = " + str(min_df))
    print("Extracting the top " + str(max_features) + " features...")

    X_train_feats = vectorizer.fit_transform(X_train) # fitting vectorizer on training data
    X_test_feats = vectorizer.transform(X_test) # fitting vectorizer on test data

    return X_train_feats, X_test_feats, y_train, y_test, vectorizer

 
# saving the vectorizer
def save_results(vectorizer):
    outpath_vectorizer = os.path.join(os.getcwd(), "models", "mtfidf_vectorizer.joblib")

    dump(vectorizer, open(outpath_vectorizer, 'wb') )

    print("I have saved the vectorizer in the folder ´models´")


# main function

def main():
    args = input_parse()
    X,y = load_data(args.filename)
    vectorizer = vectorize_function(X, y, args.test_size, args.random_state, tuple(args.ngram_range), args.lowercase,  args.max_df, args.min_df,args.max_features)
    save_results(vectorizer)

if __name__ == '__main__':
    main()



