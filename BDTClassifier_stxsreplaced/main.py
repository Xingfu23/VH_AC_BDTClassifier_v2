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

from tools.common_tool import get_args, colloect_samples, output_xmlfile
from tools.plot_tool import importance_plot, probability_plot, roc_curve
from tools.xgboost2tmva import convert_model

'''
command example: python3 main.py -n "XGBoost_VHMetTag_BDT_ULv1" -era "2017" -x 1
'''

def main():
    args = get_args()
    if args == 0:
        return 0

    # If the era euqals to 'total', then combine all the data from 2016, 2017 and 2018
    era_list = ['2016preVFP', '2016postVFP', '2017', '2018']
    df_bkg_tot = pd.DataFrame()
    df_sig_tot = pd.DataFrame()
    if args.era == 'total':
        for era in era_list:
            print(f"Collecting SM background and signal from {era}...")
            df_bkg = colloect_samples(0, era)
            df_sig = colloect_samples(1, era)
            if df_bkg_tot.empty and df_sig_tot.empty:
                df_bkg_tot = df_bkg.copy()
                df_sig_tot = df_sig.copy()
            else:
                df_bkg_tot = pd.concat([df_bkg_tot, df_bkg], ignore_index=True, axis=0)
                df_sig_tot = pd.concat([df_sig_tot, df_sig], ignore_index=True, axis=0)
    else:
        print(f"Collecting SM background and signal from {args.era}...")
        df_bkg_tot = colloect_samples(0, args.era)
        df_sig_tot = colloect_samples(1, args.era)
    
    # Adding a column to indicate whether the event is signal or background
    df_bkg_tot['sig/bkg'] = 0
    df_sig_tot['sig/bkg'] = 1

    # Seperate training dataset(80%) and testing dataset(10% for validation, and another 10% for testing)
    X_sig = df_sig_tot.iloc[:, :-1] # The last two columns are 'sig/bkg' and 'training_weight'
    X_bkg = df_bkg_tot.iloc[:, :-1] # The last two columns are 'sig/bkg' and 'training_weight'
    y_sig = df_sig_tot['sig/bkg']
    y_bkg = df_bkg_tot['sig/bkg']

    random_state = 69

    # Split the dataset into training, validation and testing dataset
    X_sig_train, X_sig_tmp, y_sig_train, y_sig_tmp = train_test_split(X_sig, y_sig, test_size=0.3, stratify=y_sig, random_state=random_state)
    X_sig_valid, X_sig_test, y_sig_valid, y_sig_test = train_test_split(X_sig_tmp, y_sig_tmp, test_size=0.5, stratify=y_sig_tmp, random_state=random_state)
    X_bkg_train, X_bkg_tmp, y_bkg_train, y_bkg_tmp = train_test_split(X_bkg, y_bkg, test_size=0.3, stratify=y_bkg, random_state=random_state)
    X_bkg_valid, X_bkg_test, y_bkg_valid, y_bkg_test = train_test_split(X_bkg_tmp, y_bkg_tmp, test_size=0.5, stratify=y_bkg_tmp, random_state=random_state)

    # Sum of sample weights should be modified to be the same as one in background to avoid unbalanced issue
    X_sig_train['training_weight'] = X_sig_train['training_weight'] / X_sig_train['training_weight'].sum() * X_bkg_train['training_weight'].sum()
    X_sig_valid['training_weight'] = X_sig_valid['training_weight'] / X_sig_valid['training_weight'].sum() * X_bkg_valid['training_weight'].sum()
    X_sig_test['training_weight'] = X_sig_test['training_weight'] / X_sig_test['training_weight'].sum() * X_bkg_test['training_weight'].sum()

    _X_train = pd.concat([X_sig_train, X_bkg_train], ignore_index=True, axis=0)
    _X_valid = pd.concat([X_sig_valid, X_bkg_valid], ignore_index=True, axis=0)
    _X_test = pd.concat([X_sig_test, X_bkg_test], ignore_index=True, axis=0)

    y_train = pd.concat([y_sig_train, y_bkg_train], ignore_index=True, axis=0)
    y_valid = pd.concat([y_sig_valid, y_bkg_valid], ignore_index=True, axis=0)
    y_test = pd.concat([y_sig_test, y_bkg_test], ignore_index=True, axis=0)

    X_train = _X_train.iloc[:, :-1]
    X_train_weight = abs(_X_train['training_weight'])
    X_test = _X_test.iloc[:, :-1]
    X_test_weight = abs(_X_test['training_weight'])
    X_valid = _X_valid.iloc[:, :-1]
    X_valid_weight = abs(_X_valid['training_weight'])

    eval_set = [(X_train, y_train), (X_valid, y_valid)]

    def objective(trial, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid):
        """
        A function to train a model using different hyperparamerters combinations provided by Optuna.
        """
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [150, 200, 300 ,1000, 1500, 3000]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05),
            'gamma': trial.suggest_float('gamma', 0.01, 1),
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'min_child_weight': trial.suggest_float('min_child_weight', 1., 10.),
            'max_delta_step': trial.suggest_float('max_delta_step', 1, 10.),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1, 10),
        }

        # XGBoost sklearn configuration
        _XGBEngine = xgb.XGBClassifier(
            booster='gbtree',
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=10,
            tree_method='gpu_hist', # Using GPU
            gpu_id=0,               # Using GPU
            predictor='gpu_predictor' # Using GPU
        )

        # Training
        _XGBEngine.fit(X_train, y_train, 
                       sample_weight=X_train_weight,
                       sample_weight_eval_set=[X_train_weight, X_valid_weight],
                       eval_set=eval_set, 
                       verbose=False
        )

        # Return the value of log loss
        return metrics.log_loss(y_valid, _XGBEngine.predict_proba(X_valid))

    # Creating Optuna object and minimizing the value of log loss
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=69), direction='minimize')
    study.optimize(objective, n_trials=50)

    # # Visualizing the optimization process
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.savefig(f"plots/{args.PlotName}_OptimizationHistory.png")

    # Showing optimization results
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial parameters:")
    trial = study.best_trial
    params = trial.params
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in params.items():
        print(f"    {key}: {value}")

    best_XGBEngine = xgb.XGBClassifier(
        **params,
        booster='gbtree',
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=10,
        tree_method='gpu_hist', # Using GPU
        gpu_id=0,               # Using GPU
        predictor='gpu_predictor' # Using GPU
    )

    best_XGBEngine.fit(X_train, y_train, 
                       sample_weight=X_train_weight,
                       sample_weight_eval_set=[X_train_weight, X_valid_weight],
                       eval_set=eval_set, 
                       verbose=True
    )

    # Predicting the test dataset and printing the auc score
    y_pred = pd.DataFrame(best_XGBEngine.predict(X_test))
    y_pred_prob_train = pd.DataFrame(best_XGBEngine.predict_proba(X_train))
    y_pred_prob_valid = pd.DataFrame(best_XGBEngine.predict_proba(X_valid))
    y_pred_prob_test = pd.DataFrame(best_XGBEngine.predict_proba(X_test))
    accuracy = metrics.accuracy_score(y_test, y_pred, sample_weight=X_test_weight)
    print(f"Accuracy: {accuracy:.4f}")

    # ROC curve
    # Train ROC
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, best_XGBEngine.predict_proba(X_train)[:, 1], pos_label=1, sample_weight=X_train_weight)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    # Valid ROC
    fpr_valid, tpr_valid, _ = metrics.roc_curve(y_valid, best_XGBEngine.predict_proba(X_valid)[:, 1], pos_label=1, sample_weight=X_valid_weight)
    roc_auc_valid = metrics.auc(fpr_valid, tpr_valid)
    # Test ROC
    fpr_test, tpr_test, _ = metrics.roc_curve(y_test, best_XGBEngine.predict_proba(X_test)[:, 1], pos_label=1, sample_weight=X_test_weight)
    roc_auc_test = metrics.auc(fpr_test, tpr_test)

    print(f"Train AUC: {roc_auc_train:.4f}")
    print(f"Test AUC: {roc_auc_test:.4f}")

    # Ouput xml file
    output_xmlfile(args.PlotName, best_XGBEngine, args.xmlfile)

    # Make plots
    # Check the output folder
    if not os.path.exists('output_plots'):
        os.makedirs('output_plots')
    
    # Drawing importance plot
    importance_plot(best_XGBEngine, args.PlotName)

    # Drawing probability plot
    probability_plot(y_pred_prob_test, y_test, args.PlotName)

    # Drawing ROC curve
    fpr_list = [fpr_train, fpr_valid, fpr_test]
    tpr_list = [tpr_train, tpr_valid, tpr_test]
    roc_auc_list = [roc_auc_train, roc_auc_valid, roc_auc_test]
    roc_curve(fpr_list, tpr_list, roc_auc_list, args.PlotName)

    return 0

if __name__ == "__main__":
    main()