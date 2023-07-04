import argparse
import os
import yaml
import pandas as pd
import uproot
import xgboost as xgb

from tools.bdt_vars import *
from tools.xgboost2tmva import *

def get_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--PlotName', help='Name for output plot and xml file', type=str, required=True)
    parser.add_argument('-ac', '--ac', help='The type of ac, there are 3 types: fa2, fa3, L1', type=str, required=True)
    parser.add_argument('-x', '--xmlfile', help='Output xml file or not', type=bool, default=False)

    args = parser.parse_args()
    return args

def file_exsit(file_path:str)->bool:
    try:
        os.path.exists(file_path)
    except:
        print(f"The file {file_path} does not exist.\n")
        print(f"Please check the file path and try again.\n")
        return False
    return True

def colloect_samples(mc_type:int, ac_type:str=None)->pd.DataFrame:
    df_tot = pd.DataFrame()

    # Loading files, the list comes from background part of 'importfiles.yaml'
    with open('tools/importfiles.yaml', 'r') as _import_f:
        import_f = yaml.safe_load(_import_f)
    era_list = ['2018', '2017', '2016preVFP', '2016postVFP']
    for era_entry in range(len(era_list)):
        for file_entry in range(2): # Each era got zh and wh 2 channels
            if mc_type == 0: # background
                fileroute = import_f['path'][0] + import_f['background'][era_list[era_entry]][file_entry]
                import_display = import_f['background'][era_list[era_entry]][file_entry]
            else: # signal
                fileroute = import_f['path'][0] + import_f['signal'][ac_type][era_list[era_entry]][file_entry]
                import_display = import_f['signal'][ac_type][era_list[era_entry]][file_entry]
            # To check the existance of target file.
            try:
                os.path.exists(fileroute)
            except:
                print(f"The file {fileroute} does not exist.\n")
                print(f"Please check the file path and try again.\n")
                continue
            print(f"Importing file:\n {import_display}...")

            # target tree location and take feature varibales
            file = uproot.open(fileroute)
            tree_loc = file.keys()[-1]
            tree = file[tree_loc]
            df_single = tree.arrays(training_dataset, library="pd")
            if df_tot.empty:
                df_tot = df_single.copy()
            else:
                df_tot = pd.concat([df_tot, df_single], axis=0).reset_index(drop=True)
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