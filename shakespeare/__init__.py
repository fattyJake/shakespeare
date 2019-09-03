#-*- coding: utf-8 -*-
###############################################################################
#          _____ _           _                                                #
#         /  ___| |         | |                                               #
#         \ `--.| |__   __ _| | _____  ___ _ __   ___  __ _ _ __ ___          #
#          `--. \ '_ \ / _` | |/ / _ \/ __| '_ \ / _ \/ _` | '__/ _ \         #
#         /\__/ / | | | (_| |   <  __/\__ \ |_) |  __/ (_| | | |  __/         #
#         \____/|_| |_|\__,_|_|\_\___||___/ .__/ \___|\__,_|_|  \___|         #
#                                         | |                                 #
#                                         |_|                                 #
#                                                                             #
###############################################################################
# A python package designed to detect gaps using the CARA conditions
# NOTE: this model is NOT designed for prospective advantage
#
# Authors:  William Kinsman, Chenyu (Oliver) Ha, Harshal Samant, Yage Wang
# Created:  11.02.2017
# Version:  2.5.0
###############################################################################


# check for dependancies on import
import os
from pkgutil import iter_modules

def module_exists(module_name):
    return module_name in (name for loader, name, ispkg in iter_modules())
for module in ['scipy','numpy','pandas','matplotlib','sklearn','pyodbc','xgboost']:
    if not module_exists(module):
        raise ImportError('\nRequired shakespeare dependencies not detected. Please install: ' + ', '.join(module))

mingw_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), r'mingw64', r'bin')
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import re
import pickle
import itertools
import operator
from collections import Counter
import gc
from datetime import datetime
import pyodbc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle

from . import fetch_db
from . import vectorizers
from . import visualizations
from . import xgb_coef

def detect(payer,server,memberID_list=None,date_start=None,date_end=None,file_date_lmt=None,
           mem_date_start=None,mem_date_end=None,model=63,auto_update=False,threshold=0,
           output_path=None,get_indicators=False,top_n_indicator=5):
    """
    Detects the HCCs a patient may have, and supporting evidence

    Parameters
    --------
    payer : str
        name of payer table (e.g. 'CD_HEALTHFIRST')
        
    server : str
        CARA server on which the payer is located ('CARABWDB03')
        
    memberID_list : list, optional (default: None)
        list of memberIDs (e.g. [1120565]); if None, get results from all members under payer
        
    date_start : str, optional (default: None)
        string as 'YYYY-MM-DD' to get claims data from
        
    date_end : str, optional (default: None)
        string as 'YYYY-MM-DD' to get claims data to
        
    file_date_lmt : str, optional (default: None)
     string as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
     
    mem_date_start : str, optional (default: None)
        as 'YYYY-MM-DD', starting date to filter memberIDs by dateInserted
        
    mem_date_end : str, optional (default: None)
        as 'YYYY-MM-DD', ending date to filter memberIDs by dateInserted
        
    model : int, optional (default: 63)
        an integer of model version ID in MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only support 'CDPS', 'CMS' and 'HHS'
        
    auto_update : boolean, optional (default: False)
        if True, and no matching model pickled, it will call `update` function first
        
    threshold : float, optional (default: 0)
        a float between 0 and 1 for filtering output confidence above it
        
    output_path : str, optional (default: None)
        if not None, provided with the path of output EXCEL sheet to store HCC probs, financial values for all member list
        
    get_indicators : boolean, optional (default: False)
        if False, only return probabilities for each HCC each member; if True, add supporting evidences of each member to each HCC
        
    top_n_indicator : int, optional (default: 5)
        how many indicators to output for each member each HCC
    
    Return
    --------
    member's condition:
        [(memberID, HCC, prob, tp_flag)]
        OR
        [(memberID, HCC, prob, tp_flag, top_n_codes, top_n_encounter_id, top_n_coefficient)]

    Examples
    --------
    >>> from shakespeare import detect
    >>> detect(payer="CD_GATEWAY", server="CARABWDB06", date_start='2017-01-01', date_end='2017-12-31', threshold=0.1, top_n_indicator=5, get_indicators=True)
        [(1874863, 'HCC100', 0.6826, False,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-I679', 'ICD10-R0989'],
          [260407786, 238479950, 261261633, 263391390, 260296947],
          [0.014031, 0.011159, 0.008204, 0.003258, 0.002997]),
         (1875152, 'HCC100', 0.7459, False,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-G459', 'CPT-70496'],
          [244335010, 245688342, 245688341, 251636401, 246095758],
          [0.014031,  0.011159, 0.008204, 0.004636, 0.002898]),
         ...
         (2002308, 'HCC114', 0.131, False,
          ['ICD10-J189', 'CPT-99233', 'CPT-99232', 'CPT-99223', 'ICD10-R918'],
          [264730331, 264730331, 264787243, 265172978, 265317623],
          [0.031377, 0.012256, 0.009105, 0.003366, 0.001848]),
         (2002971, 'HCC114', 0.1319, False,
          ['ICD10-J189', 'CPT-99291', 'CPT-99232', 'ICD10-J90', 'CPT-99223'],
          [265224944, 265150570, 265367997, 264964282, 264567584],
          [0.031377, 0.009218, 0.009105, 0.0043, 0.003366])]
    """

    # initialize
    if memberID_list and not isinstance(memberID_list, list): memberID_list = [memberID_list]
    if not memberID_list: memberID_list = list({tup[0] for tup in fetch_db.get_members(payer, server, mem_date_start, mem_date_end)})
    memberID_list = shuffle(memberID_list)

    if date_end: year = int(date_end[:4])
    else: year = str(datetime.now().year)

    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model))):
        if auto_update:
            print('Model version not exist yet! Updating...')
            update(model, year)
        else: raise FileNotFoundError(f'Model version not exist! Please use `shakespeare.update({model}, {year})` to train ML for Model {model}.')

    print("Getting data...")
    if len(memberID_list) <= 40000:
        table = fetch_db.batch_member_codes(payer=payer, server=server, memberIDs=memberID_list,
                                            date_start=date_start, date_end=date_end, file_date_lmt=file_date_lmt,
                                            mem_date_start=mem_date_start, mem_date_end=mem_date_end, model=model,
                                            get_client_id=False)
        if not table: return []
        return detect_members(table, model, auto_update, threshold, output_path, get_indicators, top_n_indicator)
    else:
        memberID_list = [memberID_list[i:i + 40000] for i in range(0, len(memberID_list), 40000)]
        results = []
        print(f"Total batches: {len(memberID_list)}")
        for batch, sub_mem in enumerate(memberID_list):
            print(f"###################### Processing batch {batch+1} ######################")
            table = fetch_db.batch_member_codes(payer=payer, server=server, memberIDs=sub_mem,
                                                date_start=date_start, date_end=date_end, file_date_lmt=file_date_lmt,
                                                mem_date_start=mem_date_start, mem_date_end=mem_date_end, model=model,
                                                get_client_id=False)
            if not table: continue
            results.extend(detect_members(table, model, auto_update, threshold, output_path, get_indicators, top_n_indicator))
            del table
            gc.collect()
            
        return results

def detect_members(table, model=63, auto_update=False, threshold=0, output_path=None,
                   get_indicators=False, top_n_indicator=5):
    """
    Detects the HCCs a patient may have, and supporting evidence

    Parameters
    --------
    table : list of tuples
        list of tuples of format [(mem_id, enc_id, code)]; e.g. [(1120565, '130008347', 'ICD9-4011'), (1120565, '130008347', 'CPT-73562')]
        
    model : int, optional (default: 63)
        an integer of model version ID in MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only support 'CDPS', 'CMS' and 'HHS'
        
    auto_update : boolean, optional (default: False)
        if True, and no matching model pickled, it will call `update` function first
        
    threshold : float, optional (default: 0)
        a float between 0 and 1 for filtering output confidence above it
        
    output_path : str, optional (default: None)
        if not None, provided with the path of output EXCEL sheet to store HCC probs, financial values for all member list
        
    get_indicators : boolean, optional (default: False)
        if False, only return probabilities for each HCC each member; if True, add supporting evidences of each member to each HCC
        
    top_n_indicator : int, optional (default: 5)
        how many indicators to output for each member each HCC
    
    Return
    --------
    member's condition:
        [(memberID, HCC, prob, norm_prob, tp_flag)]
        OR
        [(memberID, HCC, prob, norm_prob, tp_flag, top_n_codes, top_n_encounter_id, top_n_coefficient)]

    Examples
    --------
    >>> from shakespeare import detect_members
    >>> detect_members(table, threshold=0.1, top_n_indicator=5, get_indicators=True)
        [(1874863, 'HCC100', 0.6826, False,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-I679', 'ICD10-R0989'],
          [260407786, 238479950, 261261633, 263391390, 260296947],
          [0.014031, 0.011159, 0.008204, 0.003258, 0.002997]),
         (1875152, 'HCC100', 0.7459, False,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-G459', 'CPT-70496'],
          [244335010, 245688342, 245688341, 251636401, 246095758],
          [0.014031,  0.011159, 0.008204, 0.004636, 0.002898]),
         ...
         (2002308, 'HCC114', 0.131, False,
          ['ICD10-J189', 'CPT-99233', 'CPT-99232', 'CPT-99223', 'ICD10-R918'],
          [264730331, 264730331, 264787243, 265172978, 265317623],
          [0.031377, 0.012256, 0.009105, 0.003366, 0.001848]),
         (2002971, 'HCC114', 0.1319, False,
          ['ICD10-J189', 'CPT-99291', 'CPT-99232', 'ICD10-J90', 'CPT-99223'],
          [265224944, 265150570, 265367997, 264964282, 264567584],
          [0.031377, 0.009218, 0.009105, 0.004300, 0.003366])]
    """
    print('Start: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model))):
        if auto_update:
            print('Model version not exist !')
            update(model, datetime.now().year)
        else: raise ValueError('Model version not exist !')
    
    variables  = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'variables','variables_{}'.format(model)),"rb"))
    indicators = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'indicators','indicators_{}'.format(model)),"rb"))
    mappings   = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'mappings','mapping_{}'.format(model)),'rb'))
    ensemble   = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model)),"rb"))

    df_member = _create_df(table)
    del table
    gc.collect()

    # check for direct mappings
    member_known = {}
    for idx, row in df_member.iterrows():
        knowns = []
        for i in row['CODE']:
            if i in mappings and i not in knowns: knowns.extend(mappings[i])
        member_known[row['MEMBER']] = list(set(knowns))

#    # Don't miss members without any claims
#    all_members      = dict(fetch_db.get_members(payer, server, mem_date_start, mem_date_end))
#    all_members      = {i: all_members[i] for i in memberID_list}
#    existing_members = list(df_member['MEMBER'])
#    other_members    = [(i, all_members[i])for i in list(set(all_members)-set(existing_members))]
#    if other_members:
#        df_other         = pd.DataFrame(other_members)
#        df_other.columns = ['MEMBER','MEMBER_CLIENT_ID']
#        df_other['CODE'] = [[] for _ in range(len(df_other))]
#        df_other['ENCOUNTER_ID'] = [[] for _ in range(len(df_other))]
#        df_member        = pd.concat([df_member, df_other], ignore_index=True, sort=False)
#    for ID in all_members:
#        if ID not in member_known: member_known[ID] = []
    print("Total members with codes: {}".format(str(df_member.shape[0])))

    #cached_codes = {row['MEMBER']:row['CODE'] for _, row in df_member.iterrows()}

    # Vectorizing
    print("Vectoring...")
    df_vector = df_member[['MEMBER', 'CODE']].copy()
    df_vector['temp'] = np.nan
    df_vector['temp'] = df_vector['temp'].astype(object)
    
    for idx, row in df_vector.iterrows():
        vector = vectorizers.build_member_input_vector(row['CODE'], variables)
        df_vector.at[idx, 'temp'] = vector
    del df_vector['CODE']

    print("Running ML...")
    input_data = csr_matrix(vstack(df_vector['temp'].tolist()))
    for k, v in ensemble.items():
        if v['classifier']:
            scores = v['classifier'].predict_proba(input_data)[:, 1]
            df_vector = pd.concat([df_vector, pd.Series(scores, name=k)], axis=1)
    del df_vector['temp']
    gc.collect()

    # modify knowns
#    for index, value in df_member.iterrows():
#        for col in member_known[value['MEMBER']]: df_vector.at[index, col] = 0.0
    df_vector = df_vector.dropna(axis=1, how='any')

    if output_path:
        print("Saving results...")
        # df_vector = df_vector.sort_values('PRED_VALUE', ascending=False)
        df_vector.to_excel(output_path, sheet_name='output', header=True, index=False)
    
    df_condition = pd.melt(df_vector, id_vars=['MEMBER'], value_vars=[i for i in df_vector.columns if i.startswith('HCC')])
    df_condition.columns = ['MEMBER', 'HCC', 'CONFIDENCE']
    df_condition['CONFIDENCE_NORM'] = df_condition['CONFIDENCE'].map(lambda x: np.tanh(np.power(x, 1/2)*4) if x < 0.012091892 else np.power(x, 1/5))
    df_condition = df_condition.loc[df_condition['CONFIDENCE_NORM'] >= threshold, :]
    df_condition['CONFIDENCE'] = df_condition['CONFIDENCE'].round(4)
    df_condition['CONFIDENCE_NORM'] = df_condition['CONFIDENCE_NORM'].round(4)
    df_condition['TP_FLAG'] = df_condition[['MEMBER', 'HCC']].apply(lambda x: 1 if x[1] in member_known.get(x[0], []) else 0, axis=1)
    
    if get_indicators:
        print("Finding indicators...")
        del df_vector
        gc.collect()
        filtered_condtions = df_condition.groupby('MEMBER')['HCC'].agg(list).to_dict()
        df_member['FEATURE_IMPORTANCE'] = np.nan
        df_member['FEATURE_IMPORTANCE'] = df_member['FEATURE_IMPORTANCE'].astype(object)
        input_data = input_data.toarray()
        df_list = []
        for HCC in indicators.keys():
            if indicators[HCC] is None: continue
            df_HCC_member = df_member.copy(deep=True)
            df_HCC_member['HCC'] = HCC
            coef_matrix = (input_data * indicators[HCC]).round(6)
            for idx, member in df_HCC_member[['MEMBER']].iterrows():
                try:
                    if HCC not in filtered_condtions[member['MEMBER']]: continue
                except KeyError: continue
                i = list(np.nonzero(coef_matrix[idx])[0])
                coef_dict = dict(zip([variables[index] for index in i], list(coef_matrix[idx][i])))
                coef_dict = dict(sorted(coef_dict.items(), key=operator.itemgetter(1), reverse=True)[:top_n_indicator])
                codes = df_HCC_member.at[idx, 'CODE']
                indices = [codes.index(code) for code in coef_dict]
                df_HCC_member.at[idx, 'CODE'] = [codes[i] for i in indices]
                df_HCC_member.at[idx, 'PRA_ID'] = [df_member.at[idx, 'PRA_ID'][i] for i in indices]
                df_HCC_member.at[idx, 'FEATURE_IMPORTANCE'] = list(coef_dict.values())
            df_list.append(df_HCC_member)
        df_member = pd.concat(df_list, axis=0)
        del input_data, df_list
        gc.collect()
        df_condition = pd.merge(df_condition, df_member, on=['MEMBER', 'HCC'], how='left')
    
    print('End  : ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return [tuple(x) for x in df_condition.values]

def update(model, year_of_service):
    """
    Since CARA use different ICD-HCC mappings at different service year, this function is for updating
    mappings, variable spaces and XGBoost models for all lines of business. 

    Parameters
    --------
    model : int
        an integer of model version ID in MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only support 'CDPS', 'CMS' and 'HHS'
        
    year_of_service : int
        a intiger of the year of service for training data retrieval purpose

    Examples
    --------
    >>> from shakespeare import update
    >>> update(63, 2018)
    ############################## Training New Model 63 ##############################
    Fetching training dateset...
    Time elapase: 0:11:00.040693
    
    Updating new mapping table...
    Time elapase: 0:00:00.249601
    
    Updating new variable spaces...
    Time elapase: 0:00:12.090078
    
    Updating new machine learning models...
    Training HCC1
    Training HCC10
    ...
    Training HCC96
    Training HCC99
    Time elapase: 05:08:24.738294
    
    Updating new global supporting evidences...
    Time elapase: 0:00:06.009842
    ############################ Finished Training Model 63 ###########################
    """
    assert isinstance(model, int) and year_of_service > 2010, print('The year of service must be a integer greater than 2010')

    print('############################## Training New Model '+str(model)+' ##############################')
    training_set = _get_training_set(year_of_service, model)
    _update_mappings(model)
    _update_variables(training_set, model)
    _update_ensembles(training_set, model)
    _update_indicators(model)
    print('############################ Finished Training Model '+str(model)+' ###########################')

def delete(model):
    """
    Module for deleting old models to free up disk space

    Parameters
    --------
    model : int
        an integer of model version ID in MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only support 'CDPS', 'CMS' and 'HHS'
    
    Examples
    --------
    >>> from shakespeare import delete
    >>> delete(63)
    """
    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model))):
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model)))
    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'mappings','mapping_{}'.format(model))):
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'mappings','mapping_{}'.format(model)))
    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'variables','variables_{}'.format(model))):
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'variables','variables_{}'.format(model)))
    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'indicators','indicators_{}'.format(model))):
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'indicators','indicators_{}'.format(model)))

###########################  PRIVATE FUNCTIONS  ###############################

def _create_df(table):
    df_member = pd.DataFrame(table)
    specialists = {2, 3, 4, 5, 7, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 29, 30, 31, 35, 37, 39, 40, 41, 42, 43, 44, 45, 47,
                   50, 51, 52, 59, 60, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 82, 84, 88, 89, 90, 92, 93, 95, 100,
                   102, 103, 106, 109, 111, 113, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 133, 135, 136, 140, 142, 143, 144, 145,
                   146, 147, 148, 149, 150, 152, 153, 156, 157, 158, 159, 160, 164, 165, 166, 171, 172, 173, 174, 175, 176, 177, 179, 180,
                   181, 182, 186, 187, 188, 189, 191, 192, 195, 196, 198, 199, 202, 206, 208, 209, 210, 212, 214, 215, 217, 220, 221, 222,
                   223, 224, 228, 230, 231, 232, 233, 234, 237, 998, 999}
    
    if df_member.shape[1] == 5:
        df_member.columns = ['MEMBER','MEMBER_CLIENT_ID','PRA_ID','SPEC_ID','CODE']
        df_member['PRA_ID'] = df_member['PRA_ID'].astype(int)
        df_member = df_member.groupby(['MEMBER','MEMBER_CLIENT_ID', 'CODE'])['PRA_ID', 'SPEC_ID'].agg(list).reset_index()
        df_member['SPEC_ID'] = df_member['SPEC_ID'].map(lambda x: [True if spec in specialists else False for spec in x])
        df_member['PRA_ID'] = df_member.apply(lambda x: ','.join([str(pra) for i, pra in enumerate(x['PRA_ID']) if x['SPEC_ID'][i]]), axis=1)
        df_member = df_member.groupby(['MEMBER','MEMBER_CLIENT_ID'])['CODE', 'PRA_ID'].agg(list).reset_index()
    if df_member.shape[1] == 4:
        df_member.columns = ['MEMBER','PRA_ID','SPEC_ID','CODE']
        df_member['PRA_ID'] = df_member['PRA_ID'].astype(int)
        df_member = df_member.groupby(['MEMBER', 'CODE'])['PRA_ID', 'SPEC_ID'].agg(list).reset_index()
        df_member['SPEC_ID'] = df_member['SPEC_ID'].map(lambda x: [True if spec in specialists else False for spec in x])
        df_member['PRA_ID'] = df_member.apply(lambda x: ','.join([str(pra) for i, pra in enumerate(x['PRA_ID']) if x['SPEC_ID'][i]]), axis=1)
        df_member = df_member.groupby(['MEMBER'])['CODE', 'PRA_ID'].agg(list).reset_index()
    return df_member

def _get_coef(classifier):
    if 'XGBClassifier' in str(classifier.__str__): return xgb_coef.coef(classifier)
    if 'CalibratedClassifierCV' in str(classifier.__str__):
        coefs = [xgb_coef.coef(c.base_estimator) for c in classifier.calibrated_classifiers_]
        coefs = np.sum(coefs, axis=0) / classifier.cv
        return coefs

def _shuffle(positives, negatives, member_vectors):
    """
    Partition whole dataset randomly into training set, development set and test set (20%)
    """
    # Prepare DL dateset
    y = []
    X  = []
    for member in positives:
       X.append(csr_matrix(member_vectors[member]))
       y.append(1)

    for member in negatives:
       X.append(csr_matrix(member_vectors[member]))
       y.append(0)
    X = csr_matrix(vstack(X))

    # Split dataset
    X, y = shuffle(X, y, random_state=42)
    return X, y

def _get_training_set(year, model):
    print("Fetching training dateset...")
    start_time = datetime.now()
    p_year = str(year-1)
    codes = {}

    for y in range(year-2, year):
        for m in range(1,12):
            mem_date_start = datetime(y, m, 1).strftime('%Y-%m-%d')
            mem_date_end   = datetime(y, m+1, 1).strftime('%Y-%m-%d')
            sub_codes = fetch_db.batch_member_codes(payer='CD_HEALTHFIRST', server='CARABWDB03', date_start=p_year+'-01-01', date_end=p_year+'-12-31',
                                                    mem_date_start=mem_date_start, mem_date_end=mem_date_end, model=model, get_client_id=False)
            sub_codes = pd.DataFrame(sub_codes)
            sub_codes.columns = ['mem_id', 'enc_id', 'code']
            sub_codes = sub_codes.groupby('mem_id')['code'].agg(list)
            sub_codes = sub_codes.apply(lambda x: list(set(x)))
            codes.update(dict(sub_codes))
            del sub_codes
            gc.collect()
    
    print("Time elapase: "+str(datetime.now() - start_time))
    return codes

def _get_model_name(model):
    # build model dictionary
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER=MPBWDB1;').cursor()
    model_name = cursor.execute("""
                                SELECT [mv_ModelVersionName]
                                FROM [CARA2_Controller].[dbo].[ModelVersions]
                                WHERE mv_IsActive=1 
                                AND mv_ModelVersionID = {}""".format(model)).fetchall()
    try: model_name = model_name[0][0]
    except:
        pass
        raise ValueError('{} does not have a corresponding SQL table. Not building this model.'.format(model))
    
    # get mapping of each model from respective tables
    cursor.close()
    return re.findall("""^[a-zA-Z]*""", model_name)[0].upper()

def _update_mappings(model):
    print("\nUpdating new mapping table...")
    start_time = datetime.now()
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER=MPBWDB1;').cursor()
    
    model_name = _get_model_name(model)
    if   'CMS' in model_name:
        # fetch the mapping
        mapping_dict = cursor.execute("""
            SELECT ICDVersionind,UPPER(icd_code),cmsfh_hcccode FROM [CARA2_Processor].[dbo].[ModelCmsMapHccIcd]
            WHERE mdst_ModelSubTypeID = 1 AND mv_modelversionID = {}""".format(model)).fetchall()

    elif 'HHS' in model_name:
        mapping_dict = cursor.execute("""
            SELECT icdversionind,UPPER(icd_code),hhsfh_hcccode FROM [HIX_Processor].[dbo].[ModelHHSMapHccIcd]
            WHERE mv_modelversionID = {}""".format(model)).fetchall()

    elif 'CDPS' in model_name:
        mapping_dict = cursor.execute("""
            SELECT icdversionind,UPPER(icd_code),cdps_code FROM [CARA2_Controller].[dbo].[ModelCdpsMapCdpsIcd]
            WHERE mv_modelversionID = {}""".format(model)).fetchall()

    else: raise ValueError('The package only support CMS, HHS and CDPS, got {} instead.'.format(model_name))
    
    if not mapping_dict: raise ValueError("Empty mapping table for {}, aborting training session.".format(model))
    
    unique_codes = sorted(list({'ICD'+str(i[0])+'-'+str(i[1]) for i in mapping_dict}))
    new_mapping_dict = {i:[] for i in unique_codes}
    if 'CDPS' in model_name:
        for i in mapping_dict: new_mapping_dict['ICD'+str(i[0])+'-'+str(i[1])].append(str(i[2]))
    else:
        for i in mapping_dict: new_mapping_dict['ICD'+str(i[0])+'-'+str(i[1])].append('HCC'+str(i[2]))
    
    # dump the models
    pickle.dump(new_mapping_dict, open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'mappings','mapping_{}'.format(model)), 'wb'))
    
    print("Time elapase: "+str(datetime.now() - start_time))
    
def _update_variables(codes, model):
    print("\nUpdating new variable spaces...")
    start_time = datetime.now()
    mappings = list(pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'mappings','mapping_{}'.format(model)),"rb")))
    
    all_codes = []
    for k,v in codes.items(): all_codes.extend(v)
    freq = dict(Counter(all_codes))
    freq = {k:v for k,v in freq.items() if k not in mappings}
    
    temp = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)[:15000]
    variables = [i[0] for i in temp]
    
    pickle.dump(variables, open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'variables','variables_{}'.format(model)),"wb"))
    
    del all_codes, freq
    gc.collect()
        
    print("Time elapase: "+str(datetime.now() - start_time))

def _update_ensembles(member_codes, model):
    print("\nUpdating new machine learning models...")
    start_time = datetime.now()
    
    param = {'learning_rate':0.1, 'n_estimators ':500, 'max_depth':7, 'min_child_weight':1,
             'gamma':0, 'objective':'binary:logistic', 'subsample':0.8, 'colsample_bytree':0.6,
             'max_delta_step':1, 'n_jobs':-1, 'tree_method':'hist', 'max_bin':512, 'grow_policy':'lossguide'}

    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model))):
        os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model)))

    mapping   = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'mappings','mapping_{}'.format(model)),"rb"))
    HCCs      = list(set(itertools.chain.from_iterable(mapping.values())))
    variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'variables','variables_{}'.format(model)), "rb")) # list vars as (codetype + '-' + evidencecode)
    member_conditions = {mem: set(itertools.chain.from_iterable([mapping[c] for c in codes if c in mapping])) for mem, codes in member_codes.items()}
    member_vectors    = {mem: vectorizers.build_member_input_vector(codes, variables) for mem, codes in member_codes.items()}

    ensemble = {k: {'classifier': None,
                    'exemplarsPOS': None,
                    'exemplarsNEG': None} for k in HCCs}
    
    # for each hcc
    for HCC in sorted(HCCs):
        try: ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model)),"rb"))
        except FileNotFoundError: pass
        if ensemble[HCC]['classifier']: continue
    
        # break up into training and testing
        positives = [k for k,v in member_conditions.items() if HCC in v]
        negatives = [k for k,v in member_conditions.items() if HCC not in v]
        if len(positives) < 500: continue
        print('Training ' + HCC)
    
        X_train, y_train = _shuffle(positives, negatives, member_vectors)

        # save information and skip classifier on low information
        pos_train_len = len([v for v in y_train if v==1])
        neg_train_len = len([v for v in y_train if v==0])
        ensemble[HCC]['exemplarsPOS'] = pos_train_len
        ensemble[HCC]['exemplarsNEG'] = neg_train_len

        # run ML algo
        ensemble[HCC]['classifier'] = CalibratedClassifierCV(XGBClassifier(**param), cv=3, method='isotonic').fit(X_train, y_train)

        # save the ensemble
        pickle.dump(ensemble, open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model)), "wb"))
        del ensemble, X_train, y_train
        gc.collect()
 
    del member_conditions, member_vectors, variables
    gc.collect()
    
    print("Time elapase: "+str(datetime.now() - start_time))

def _update_indicators(model):
    print("\nUpdating new global supporting evidences...")
    start_time = datetime.now()
    
    clf = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'ensembles','ensemble_{}'.format(model)),"rb"))
    coefs = {HCC:_get_coef(c['classifier']) for HCC, c in clf.items()}
    pickle.dump(coefs, open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files',r'indicators','indicators_{}'.format(model)),"wb"))

    print("Time elapase: "+str(datetime.now() - start_time))