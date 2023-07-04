import argparse
import os
import yaml
import pandas as pd
import uproot
import xgboost as xgb

from tools.bdt_vars import *
from tools.xgboost2tmva import convert_model

def get_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--PlotName', help='Name for output plot and xml file', type=str, required=True)
    parser.add_argument('-era', '--era', help='The era of data, there are 5 types: 2018, 2017, 2016preVFP, 2016postVFP or total', type=str, required=True)
    parser.add_argument('-x', '--xmlfile', help='Output xml file or not', type=bool, default=False)

    args = parser.parse_args()

    # Check args
    if args.PlotName == None:
        print("Please give a name for output plot and era of data.\n")
        return 0
    era_list = ['2016preVFP', '2016postVFP', '2017', '2018', 'total']
    if args.era not in era_list:
        print(f"Please choose the era of data from {era_list}\n")
        return 0
    if args.xmlfile == None:
        print("There will be no xml file output.\n")
        print("If you want to output xml file, please set the value of '-x' to True.\n")

    return args

def file_exsit(file_path:str)->bool:
    try:
        os.path.exists(file_path)
    except:
        print(f"The file {file_path} does not exist.\n")
        print(f"Please check the file path and try again.\n")
        return False
    return True

def colloect_samples(_mc_type:int, era:str)->pd.DataFrame:
    df_tot = pd.DataFrame()
    UL16Lumi, UL17Lumi, UL18Lumi = 35.88, 41.48, 59.69
    sf_diphoton = 1.81
    sf_gjet    = 1.02

    # Loading files, the list comes from background part of 'importfiles.yaml'
    with open('tools/importfiles.yaml', 'r') as _import_f: 
        import_f = yaml.safe_load(_import_f)
        if _mc_type == 0: # background
            mc_type = 'background'
        else: # signal
            mc_type = 'signal'
        for file_entry in range(len(import_f[mc_type][era])):
            fileroute = import_f['path'][0] + import_f[mc_type][era][file_entry]
            import_display = import_f[mc_type][era][file_entry]

            print(f"Importing file:\n {import_display}...")

            # Check the existance of target file, if not, terminate the program
            if not file_exsit(fileroute):
                return 0

            # target tree location and take feature varibales
            file = uproot.open(fileroute)
            tree_loc = file.keys()[-1]
            tree = file[tree_loc]
            df_single = tree.arrays(training_dataset, library="pd")

            # Here we need to add a new column called 'training_weight'
            # Except gjet and diphoton samples, all the other samples 'training_weight' is 'weight' * Luminosity
            # For gjet, the value is 'preevent_weight' *  'sf_gjet' and for the diphoton, the value is 'weight' * Luminosity * 'sf_diphoton'
            
            # Check the era of data and decide the luminosity
            if era == '2016preVFP' or '2016postVFP':
                lumi = UL16Lumi
            elif era == '2017':
                lumi = UL17Lumi
            elif era == '2018':
                lumi = UL18Lumi
            else:
                print("Please check the era of data.\n")
                return 0
        
            if 'gjet' in fileroute:
                df_single['training_weight'] = df_single['preevent_weight'] * sf_gjet
            elif 'diphoton' in fileroute:
                df_single['training_weight'] = df_single['weight'] * lumi * sf_diphoton
            else:
                df_single['training_weight'] = df_single['weight'] * lumi
            
            # Remove the 'weight' and 'preevent_weight' column in origional dataframe
            df_single = df_single.drop(['weight', 'preevent_weight'], axis=1)

            if df_tot.empty:
                df_tot = df_single.copy()
            else:
                df_tot = pd.concat([df_tot, df_single], ignore_index=True, axis=0)
    return df_tot

def output_xmlfile(outputxml_name:str, XGBEngine, xmlfile:bool=False):
    if xmlfile:
        best_model = XGBEngine.get_booster().get_dump()
        # Check is there a floder named 'output_xmlfiles', 
        # if not, create one.
        if not os.path.exists('output_xmlfiles'):
            os.makedirs('output_xmlfiles')
        outputxml_path = 'output_xmlfiles/' + outputxml_name + '.xml'
        convert_model(
            best_model,
            input_variables=training_dataset_forxml,
            output_xml=outputxml_path,
        )
        print(f"Output xml file: {outputxml_path}\n")
    else:
        print("No xml file output.\n")