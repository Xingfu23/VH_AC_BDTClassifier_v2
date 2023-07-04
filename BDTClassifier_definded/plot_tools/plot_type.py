import os
import yaml
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import uproot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import xgboost as xgb

def model_importance_plot(model, plotname:str)->None:
    fig, ax = plt.subplots(figsize=(8, 6))
    featvar_im = model.feature_importances_
    xgb.plot_importance(model, ax = ax, height = 0.8, importance_type = 'weight', show_values=False)
    plt.savefig(f"output_plots/{plotname}_importance.pdf", bbox_inches='tight')
    print(f"output_plots/{plotname}_importance.pdf has been created.")
    plt.clf()

def probability_plot(y_pred_prob_test, y_test, plotname:str, ac_type:str)->None:
    output_folder = "output_plots"
    y_pred_com = pd.concat([y_pred_prob_test, y_test.reset_index(drop=True)], axis=1).dropna()
    df_histsig = y_pred_com[y_pred_com['sig/bkg'] == 1]
    df_histbkg = y_pred_com[y_pred_com['sig/bkg'] == 0]

    plt.figure(figsize=(8,6))
    ax = plt.gca()
    if ac_type == 'fa3':
        labels = [r"$f_{a1}=1.0$ SM CP-even", r"$f_{a3}=1.0$ CP-odd"]
    elif ac_type == 'fa2':
        labels = [r"$f_{a1}=1.0$ SM", r"$f_{a2}=1.0$"]
    else:
        labels = [r"$f_{a1}=1.0$ SM", r"$f_{\Lambda1}=1.0$"]
    bins = np.linspace(0., 1., 50)
    plt.hist(df_histbkg[1], bins, density=True, alpha=0.7, color='b', label=labels[0], log=False)
    plt.hist(df_histsig[1], bins, density=True, alpha=0.7, color='r', label=labels[1], log=False)
    ax.set_xlabel("Probability", fontsize=14, fontweight ='bold', loc='right')
    ax.set_ylabel("1/Events", fontsize=14, fontweight ='bold', loc='top')
    ax.set_ylim(0.05, 10)
    plt.legend(bbox_to_anchor=(1, 1), prop={'size': 12})
    plt.savefig(f"{output_folder}/probability_{plotname}.pdf", bbox_inches='tight')
    print(f"{output_folder}/probability_{plotname}.pdf has been created.")
    plt.clf()

def roc_curve(y_value:dict, plotname:str)->None:
    output_folder = "output_plots"
    # Check the input y_value should be a dictionary including train, valid, test
    if len(y_value) != 3:
        print("The input y_value should be a dictionary including train, valid, test.")
        return
    # Train ROC
    fpr_train, tpr_train, _ = metrics.roc_curve(y_value['Train'][0].values, y_value['Train'][1].values[:, 1], pos_label=1)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    # Valid ROC
    fpr_valid, tpr_valid, _ = metrics.roc_curve(y_value['Valid'][0].values, y_value['Valid'][1].values[:, 1], pos_label=1)
    roc_auc_valid = metrics.auc(fpr_valid, tpr_valid)
    # Test ROC
    fpr_test, tpr_test, _ = metrics.roc_curve(y_value['Test'][0].values, y_value['Test'][1].values[:, 1], pos_label=1)
    roc_auc_test = metrics.auc(fpr_test, tpr_test)

    plt.figure(figsize=(8,6))
    ax = plt.gca()
    plt.plot(fpr_train, tpr_train, '#ff433d', label=f"AUC(Train) = {roc_auc_train:.3f}")
    plt.plot(fpr_valid, tpr_valid, '#4af6c3', label=f"AUC(Valid) = {roc_auc_valid:.3f}")
    plt.plot(fpr_test, tpr_test, '#0068ff', label=f"AUC(Test) = {roc_auc_test:.3f}")
    plt.legend(bbox_to_anchor=(1, 0.32))
    plt.plot([0, 1], [0, 1], '--')
    ax.set_xlabel("Background Efficiency", fontsize=14, fontweight ='bold', loc='right')
    ax.set_ylabel("Signal Efficiency", fontsize=14, fontweight ='bold', loc='top')
    plt.savefig(f"{output_folder}/roc_curve_{plotname}.pdf", bbox_inches='tight')
    print(f"{output_folder}/roc_curve_{plotname}.pdf has been created.")
    plt.clf()