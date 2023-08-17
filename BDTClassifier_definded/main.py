import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import xgboost as xgb
import optuna

from tools.xgboost2tmva import *
from tools.common_tool import *
from tools.bdt_vars import *
from plot_tools.plot_type import model_importance_plot, probability_plot, roc_curve

'''
command example: python3 main.py -n "acbdt_fa31d0" -ac "fa3" -x 1
'''


def main():
    args = get_args()
    
    # Check args
    if args.PlotName == None:
        print("Please give a name for output plot and xml file.")
        return 0
    ac_type_option = ['fa2', 'fa3', 'L1']
    if args.ac not in ac_type_option:
        print(f"Please choose the type of ac from {ac_type_option}")
        return 0

    # Collect background (SM VH) and signal (AC VH) samples
    # And combine each of them into a single dataframe
    df_bkg = colloect_samples(0)
    df_sig = colloect_samples(1, args.ac)

    # Mark the background and signal variables and caluate the entry number of each
    df_bkg['sig/bkg'] = 0
    df_sig['sig/bkg'] = 1
    postive_ratio = len(df_bkg)/len(df_sig)
    print(f"Background/Signal ratio: {postive_ratio:.4f}\n")

    # Combine background and signal dataframe
    df_photondata = pd.concat([df_bkg, df_sig], ignore_index=True, axis=0)

    # Seperate training dataset(60%) and testing dataset(20% for validation, and another 20% for testing)
    X, y = df_photondata.iloc[:, :-1], df_photondata['sig/bkg']
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)
    X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=69)
    eval_set = [(X_train, y_train), (X_valid, y_valid)]

    print (f"Training dataset size: {X_train.shape}")
    print (f"Validation dataset size: {X_valid.shape}")
    print (f"Testing dataset size: {X_test.shape}")

    def objective(trial, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid):
        """
        A function to train a model using different hyperparamerters combinations provided by Optuna.
        """

        params = {
            'max_depth' : trial.suggest_int('max_depth', 2, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'gamma' : trial.suggest_float('gamma', 0.01, 1.0),
            'subsample' : trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha' : trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda' : trial.suggest_float('reg_lambda', 0.01, 1.0),
            'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.1)
        }

        # XGBoost sklearn configuration
        _XGBEngine = xgb.XGBClassifier (
            booster = 'gbtree',
            objective = 'binary:logistic',
            use_label_encoder = None,
            eval_metric = ['logloss'],
            early_stopping_rounds = 50,
            scale_pos_weight = postive_ratio,
            tree_method = 'gpu_hist', # Using GPU
            gpu_id = 0, # Using GPU
            predictor = 'gpu_predictor' # Using GPU    
        )

        # Training
        _XGBEngine.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        return metrics.log_loss(y_valid, _XGBEngine.predict_proba(X_valid))
    

    # params = {
    #     'max_depth' : [5],
    #     'min_child_weight': [3],
    #     'n_estimators' : [500],
    #     'gamma' : [0.7],
    #     'subsample' : [0.7],
    #     'colsample_bytree' : [0.8],
    #     'reg_alpha' : [0.05],
    #     'reg_lambda' : [100],
    #     'learning_rate' : [0.01, 0.005]
    # }

    # XGBEngine = GridSearchCV(
    #             estimator  = _XGBEngine,
    #             param_grid = params,
    #             scoring    = 'neg_mean_squared_error',
    #             verbose    = 3
    # )

    # Creating Optuna object and defining its parameters
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=69), direction='minimize')
    study.optimize(objective, n_trials=20)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial parameters: {study.best_trial.params}")
    print(f"Best score: {study.best_value}")

    trial = study.best_trial
    params = trial.params

    # Saving the best model as txt file
    with open(f'output_plots/{args.PlotName}.txt', 'w') as file:
        file.write(f"Best hyperparameters: {params}\n")
        file.write(f"Best score: {study.best_value}\n")
    print(f"The best hyperparameters result has been saved as 'output_plots/{args.PlotName}.txt'\n")

    best_XGBEngine = xgb.XGBClassifier(
        **params,
        objective = 'binary:logistic',
        use_label_encoder = None,
        eval_metric = ['logloss'],
        early_stopping_rounds = 10,
        tree_method='gpu_hist', 
        gpu_id=0
    )

    best_XGBEngine.fit( X_train, y_train, eval_set=eval_set, verbose=True)

    # Output xml file
    output_xmlfile(args.PlotName, best_XGBEngine, args.xmlfile)

    # Make prediction
    y_pred = pd.DataFrame(best_XGBEngine.predict(X_test), columns=['sig/bkg'])
    y_pred_prob_train = pd.DataFrame(best_XGBEngine.predict_proba(X_train))
    y_pred_prob_valid = pd.DataFrame(best_XGBEngine.predict_proba(X_valid))
    y_pred_prob_test = pd.DataFrame(best_XGBEngine.predict_proba(X_test))

    print(" ")
    print(f'Train group: {best_XGBEngine.score(X_train,y_train):.4f}')
    print(f'Test group: {best_XGBEngine.score(X_test,y_test):.4f}')

    # Make Plots
    # Check the output folder
    if not os.path.exists('output_plots'):
        os.makedirs('output_plots')
    
    # Drawing importance plot
    model_importance_plot(best_XGBEngine, args.PlotName)

    # Make probabilities histograms
    probability_plot(y_pred_prob_test, y_test, args.PlotName, args.ac)

    # Plot ROC curve
    y_valid_dict = {'Train': [y_train, y_pred_prob_train], 'Valid': [y_valid, y_pred_prob_valid], 'Test': [y_test, y_pred_prob_test]}
    roc_curve(y_valid_dict, args.PlotName)

    return 0

if __name__ == "__main__":
    main()