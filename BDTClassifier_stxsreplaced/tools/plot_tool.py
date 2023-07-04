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
import mplhep as hep

def importance_plot(model, plotname:str)->None:
    fig, ax = plt.subplots(figsize=(8, 6))
    featvar_im = model.feature_importances_
    xgb.plot_importance(model, ax = ax, height = 0.8, importance_type = 'weight', show_values=False)
    plt.savefig(f"output_plots/{plotname}_importance.pdf", bbox_inches='tight')
    print(f"output_plots/{plotname}_importance.pdf has been created.")
    plt.clf()

def probability_plot(y_pred_prob_test, y_test, plotname:str)->None:
    # Plot style
    plt.style.use(hep.style.CMS)
    hep.cms.label(loc=0, lumi=41.48, fontsize=14, year='2017')

    output_folder = "output_plots"
    y_pred_com = pd.concat([y_pred_prob_test, y_test.reset_index(drop=True)], axis=1).dropna()
    df_histsig = y_pred_com[y_pred_com['sig/bkg'] == 1]
    df_histbkg = y_pred_com[y_pred_com['sig/bkg'] == 0]

    #plt.figure(figsize=(8,6))
    ax = plt.gca()
    labels = ['background', 'signal']
    bins = np.linspace(0., 1., 50)
    plt.hist(df_histbkg[1], bins, density=True, alpha=0.7, color='b', label=labels[0], log=False)
    plt.hist(df_histsig[1], bins, density=True, alpha=0.7, color='r', label=labels[1], log=False)
    ax.set_xlabel("Probability", fontsize=14, fontweight ='bold', loc='right')
    ax.set_ylabel("1/Events", fontsize=14, fontweight ='bold', loc='top')
    plt.legend(prop={'size': 14}, loc='upper left')
    plt.savefig(f"{output_folder}/probability_{plotname}.pdf", bbox_inches='tight')
    print(f"{output_folder}/probability_{plotname}.pdf has been created.")
    plt.clf()

def roc_curve(fpr_list:list, tpr_list:list, roc_auc_list:list, plotname:str)->None:
    plt.style.use(hep.style.CMS)
    hep.cms.label(loc=0, lumi=41.48, fontsize=14, year='2017')

    output_folder = "output_plots"

    fpr_train, fpr_valid, fpr_test = fpr_list[0], fpr_list[1], fpr_list[2]
    tpr_train, tpr_valid, tpr_test = tpr_list[0], tpr_list[1], tpr_list[2]
    roc_auc_train, roc_auc_valid, roc_auc_test = roc_auc_list[0], roc_auc_list[1], roc_auc_list[2]

    ax = plt.gca()
    plt.plot(fpr_train, tpr_train, '#ff433d', label=f"AUC(Train) = {roc_auc_train:.3f}")
    plt.plot(fpr_valid, tpr_valid, '#4af6c3', label=f"AUC(Valid) = {roc_auc_valid:.3f}")
    plt.plot(fpr_test, tpr_test, '#0068ff', label=f"AUC(Test) = {roc_auc_test:.3f}")
    plt.legend(bbox_to_anchor=(1, 0.32), prop={'size': 14})
    plt.plot([0, 1], [0, 1], '--')
    ax.set_xlabel("Background Efficiency", fontsize=14, fontweight ='bold', loc='right')
    ax.set_ylabel("Signal Efficiency", fontsize=14, fontweight ='bold', loc='top')
    plt.savefig(f"{output_folder}/roc_curve_{plotname}.pdf", bbox_inches='tight')
    print(f"{output_folder}/roc_curve_{plotname}.pdf has been created.")
    plt.clf()