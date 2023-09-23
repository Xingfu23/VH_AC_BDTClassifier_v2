#Importing dataset from sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

import xgboost as xgb

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def main():
    # Load the iris dataset
    print("Loading iris dataset...")
    iris = datasets.load_iris()
    
    # Print out the description of the dataset
    print(iris.DESCR)
    
    # Create a dataframe with the four feature variables and the target variable
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    X, y = df.iloc[:, :-1], df['target']
    
    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    params = {
        'max_depth' : 3,
        'min_child_weight': 1,
        'n_estimators' : 500,
        'gamma' : 0.5,
        'subsample' : 0.8,
        'colsample_bytree' : 0.8,
        'reg_alpha' : 0.01,
        'reg_lambda' : 0.01,
        'learning_rate' : 0.005
    }
    
    XGBEngine = xgb.XGBClassifier (
        **params,
        booster = 'gbtree',
        objective = 'multi:softmax', # multi:softmax for multiclass problem, softmax for probability distribution
        num_class = 3, # Number of classes used with multi:softmax
        eval_metric = ['mlogloss'],
        early_stopping_rounds = 10,
        tree_method = 'gpu_hist', # Using GPU
        gpu_id = 0, # Using GPU
        predictor = 'gpu_predictor' # Using GPU
    )
    
    # Training
    XGBEngine.fit(X_train, y_train, eval_set=eval_set)
    
    # Make predictions for test data
    y_pred = pd.DataFrame(XGBEngine.predict(X_test), columns=['target'])
    df_pred = pd.concat([y_pred, y_test.reset_index(drop=True)], axis=1).dropna()
    
    # Change the column name
    df_pred.columns = ['pred', 'target']
    
    y_pred_prob_test = pd.DataFrame(XGBEngine.predict_proba(X_test))
    
    print(" ")
    print(f'Train group: {XGBEngine.score(X_train,y_train):.4f}')
    print(f'Test group: {XGBEngine.score(X_test,y_test):.4f}')
    
    # Print accuracy
    print(" ")
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    # Print Probability histogram
    y_pred_com = pd.concat([y_pred_prob_test, y_test.reset_index(drop=True)], axis=1).dropna()
    y_pred_com.columns = ['0', '1', '2', 'target']
    
    y_pred_com_0 = y_pred_com[y_pred_com['target'] == 0]
    y_pred_com_1 = y_pred_com[y_pred_com['target'] == 1]
    y_pred_com_2 = y_pred_com[y_pred_com['target'] == 2]
    
    labels = ['Setosa', 'Versicolor', 'Virginica']
    label_colors = ['b', 'r', 'g']
    sub_y_pred_com = [y_pred_com_0, y_pred_com_1, y_pred_com_2]
    
    for iris_type in range(3):
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        bins = np.linspace(0., 1., 20)
        for plot_entry in range(3):
            plt.hist(sub_y_pred_com[iris_type][str(plot_entry)], bins, alpha=0.7, label=labels[plot_entry], color=label_colors[plot_entry])
        ax.set_xlabel("Probability", fontsize=14, fontweight ='bold', loc='right')
        ax.set_ylabel("1/Events", fontsize=14, fontweight ='bold', loc='top')
        plt.legend(bbox_to_anchor=(1, 1), prop={'size': 12})
        plot_name = f"probability_iris_{labels[iris_type]}.png"
        plt.savefig(plot_name, bbox_inches='tight')
        print(f"Probability histogram saved as {plot_name}")
        plt.clf()
    
if __name__ == '__main__':
    main()