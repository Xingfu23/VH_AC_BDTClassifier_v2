import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.common_tool import *

def main():

    # Check args
    args = get_args()
    if args.PlotName == None:
        print("Please give a name for output plots.")
        return 0
    ac_type_option = ['fa2', 'fa3', 'L1']
    if args.ac not in ac_type_option:
        print(f"Please choose the type of ac from {ac_type_option}")
        return 0

    # Collect background (SM VH) and signal (AC VH) samples
    if args.compare:
        df_bkg = colloect_samples(0)
        df_sig_fa2 = colloect_samples(1, 'fa2')
        df_sig_fa3 = colloect_samples(1, 'fa3')
        df_sig_L1 = colloect_samples(1, 'L1')
    else:
        df_bkg = colloect_samples(0)
        df_sig = colloect_samples(1, args.ac)


    # if there is no "plots" folder, create one
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("Create a folder named 'plots' to store output plots.")
    else:
        print("Output plots will be stored in 'plots' folder.")
    
    # Plot the distribution of each variable
    for var in dataset:
        if not args.compare:
            fig, ax = plt.subplots()
            bin_num = 50
            ax.hist(df_bkg[var], bins=bin_num, density=True, color='blue', alpha=0.7, label='bkg') 
            ax.hist(df_sig[var], bins=bin_num, density=True, color='red', alpha=0.7, label='sig')
            ax.set_xlabel(var)
            # set x range based on the variable max and min with one more bin width
            bin_width_bkg = (df_bkg[var].max()-df_bkg[var].min())/bin_num
            ax.set_xlim(min(df_bkg[var].min(), df_sig[var].min())-bin_width_bkg, max(df_bkg[var].max(), df_sig[var].max())+bin_width_bkg)
            ax.set_ylabel('1/Evts')
            ax.legend(loc='best')
            plotname = 'plots/' + args.PlotName + '_' + var + '.png'
            plt.savefig(plotname)
            print(f"The plot is saved as {plotname}.")
            plt.close()
        else:
            fig, ax = plt.subplots()
            bin_num = 50
            ax.hist(df_bkg[var], bins=bin_num, density=True, color='blue', alpha=0.7, label='bkg') 
            ax.hist(df_sig_fa2[var], bins=bin_num, density=True, color='red', alpha=0.7, label='fa2')
            ax.hist(df_sig_fa3[var], bins=bin_num, density=True, color='green', alpha=0.7, label='fa3')
            ax.hist(df_sig_L1[var], bins=bin_num, density=True, color='violet', alpha=0.7, label='L1')
            ax.set_xlabel(var)
            # set x range based on the variable max and min with one more bin width
            bin_width_bkg = (df_bkg[var].max()-df_bkg[var].min())/bin_num
            ax.set_xlim(min(df_bkg[var].min(), df_sig_fa2[var].min(), df_sig_fa3[var].min(), df_sig_L1[var].min())-bin_width_bkg, max(df_bkg[var].max(), df_sig_fa2[var].max(), df_sig_fa3[var].max(), df_sig_L1[var].max())+bin_width_bkg)
            ax.set_ylabel('1/Evts')
            ax.legend(loc='best')
            plotname = 'plots/' + args.PlotName + '_all_' + var + '.png'
            plt.savefig(plotname)
            print(f"The plot is saved as {plotname}.")
            plt.close()
               
            




if __name__ == '__main__':
    main()