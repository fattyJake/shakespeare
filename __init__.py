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
###############################################################################
# A python package designed to detect gaps using the CARA conditions
# NOTE: this model is NOT designed for prospective advantage
#
# Authors:  William Kinsman, Chenyu (Oliver) Ha, Harshal Samant, Yage Wang
# Created:  11.02.2017
# Version:  2.3.0
###############################################################################


# check for dependancies on import
import os
import imp
needed_packages = []
for i in {'scipy','numpy','pandas','matplotlib','sklearn','pyodbc','xgboost'}:
    try:    imp.find_module(i)
    except: needed_packages.append(i)
if needed_packages: raise ImportError('\nRequired shakespeare dependencies not detected. Please install:' + needed_packages)

mingw_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), r'mingw64', r'bin')
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import pickle
import operator
import gc
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
import matplotlib.pyplot as plt
from . import fetch_db
from . import vectorizers
from . import visualizations
from . import intervention
from . import xgb_coef

def detect(payer,server,memberID_list=None,date_start=None,date_end=None,file_date_lmt=None,mem_date_start=None,mem_date_end=None,mode='MA',threshold=0,output_path=None,get_indicators=False,top_n_indicator=0):
    """
    Detects the HCCs a patient may have, and supporting evidence

    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param memberID_list: list of memberIDs (e.g. [1120565]); if None, get results from all members under payer
    @param date_start: as 'YYYY-MM-DD' to get claims data from
    @param date_end: as 'YYYY-MM-DD' to get claims data to
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mem_date_start: as 'YYYY-MM-DD', starting date to filter memberIDs by dateInserted
    @param mem_date_end: as 'YYYY-MM-DD', ending date to filter memberIDs by dateInserted
    @param mode:  "MA" or "ACA"
    @param threshold: a float between 0 and 1 for filtering output confidence above it
    @param output_path: if not None, provided with the path of output EXCEL sheet to store HCC probs, financial values for all member list
    @param get_indicators: if False, only return probabilities for each HCC each member; if True, add supporting evidences of each member to each HCC
    @param top_n_indicator: how many indicators to output for each member each HCC
    
    Return
    --------
    member's condition:
        [(memberID, HCC, prob)]
        OR
        [(memberID, HCC, prob, top_n_codes, top_n_encounter_id, top_n_coefficient)]

    Examples
    --------
    >>> from shakespeare import detect
    >>> detect(payer="CD_GATEWAY", server="CARABWDB06", date_start='2017-01-01', date_end='2017-12-31', threshold=0.1, top_n_indicator=5, get_indicators=True)
        [(1874863, 'HCC100', 0.6826,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-I679', 'ICD10-R0989'],
          [260407786, 238479950, 261261633, 263391390, 260296947],
          [0.014031089842319489, 0.011159094981849194, 0.008204075507819653, 0.003257597563788295, 0.0029970938339829445]),
         (1875152, 'HCC100', 0.7459,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-G459', 'CPT-70496'],
          [244335010, 245688342, 245688341, 251636401, 246095758],
          [0.014031089842319489,  0.011159094981849194, 0.008204075507819653, 0.004636371973901987, 0.0028984546661376953]),
         ...
         (2002308, 'HCC114', 0.131,
          ['ICD10-J189', 'CPT-99233', 'CPT-99232', 'CPT-99223', 'ICD10-R918'],
          [264730331, 264730331, 264787243, 265172978, 265317623],
          [0.03137729689478874, 0.012256264686584473, 0.009104656986892223, 0.0033655648585408926, 0.0018482072046026587]),
         (2002971, 'HCC114', 0.1319,
          ['ICD10-J189', 'CPT-99291', 'CPT-99232', 'ICD10-J90', 'CPT-99223'],
          [265224944, 265150570, 265367997, 264964282, 264567584],
          [0.03137729689478874, 0.009218866936862469, 0.009104656986892223, 0.004300406668335199, 0.0033655648585408926])]
    """

    # initialize
    print('Start: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if memberID_list and not isinstance(memberID_list, list): memberID_list = [memberID_list]

    print("Getting data...")
    table = fetch_db.batch_member_codes(payer=payer,
                                        server=server,
                                        memberIDs=memberID_list,
                                        date_start=date_start,
                                        date_end=date_end,
                                        file_date_lmt=file_date_lmt,
                                        mem_date_start=mem_date_start,
                                        mem_date_end=mem_date_end,
                                        get_client_id=False)
    print("Finished...")
    if not table: return {}
    return detect_members(table, mode, threshold, output_path, get_indicators, top_n_indicator)


def detect_members(input_table, mode='MA', threshold=0, output_path=None, get_indicators=False, top_n_indicator=0):
    """
    Detects the HCCs a patient may have, and supporting evidence

    @param input_table: list of tuples of format [(mem_id, enc_id, code)]; e.g. [(1120565, '130008347', 'ICD9-4011'), (1120565, '130008347', 'CPT-73562')]
    @param mode:  "MA" or "ACA"
    @param threshold: a float between 0 and 1 for filtering output confidence above it
    @param output_path: if not None, provided with the path of output EXCEL sheet to store HCC probs, financial values for all member list
    @param get_indicators: if False, only return probabilities for each HCC each member; if True, add supporting evidences of each member to each HCC
    @param top_n_indicator: how many indicators to output for each member each HCC
    
    Return
    --------
    member's condition:
        [(memberID, HCC, prob)]
        OR
        [(memberID, HCC, prob, top_n_codes, top_n_encounter_id, top_n_coefficient)]

    Examples
    --------
    >>> from shakespeare import detect
    >>> detect(table, threshold=0.1, top_n_indicator=5, get_indicators=True)
        [(1874863, 'HCC100', 0.6826,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-I679', 'ICD10-R0989'],
          [260407786, 238479950, 261261633, 263391390, 260296947],
          [0.014031089842319489, 0.011159094981849194, 0.008204075507819653, 0.003257597563788295, 0.0029970938339829445]),
         (1875152, 'HCC100', 0.7459,
          ['ICD10-I6789', 'CPT-70450', 'CPT-70551', 'ICD10-G459', 'CPT-70496'],
          [244335010, 245688342, 245688341, 251636401, 246095758],
          [0.014031089842319489,  0.011159094981849194, 0.008204075507819653, 0.004636371973901987, 0.0028984546661376953]),
         ...
         (2002308, 'HCC114', 0.131,
          ['ICD10-J189', 'CPT-99233', 'CPT-99232', 'CPT-99223', 'ICD10-R918'],
          [264730331, 264730331, 264787243, 265172978, 265317623],
          [0.03137729689478874, 0.012256264686584473, 0.009104656986892223, 0.0033655648585408926, 0.0018482072046026587]),
         (2002971, 'HCC114', 0.1319,
          ['ICD10-J189', 'CPT-99291', 'CPT-99232', 'ICD10-J90', 'CPT-99223'],
          [265224944, 265150570, 265367997, 264964282, 264567584],
          [0.03137729689478874, 0.009218866936862469, 0.009104656986892223, 0.004300406668335199, 0.0033655648585408926])]
    """

    assert mode in ['MA', 'ACA'], print("AttributeError: mode must be either 'MA' or 'ACA', {} provided instead.".format(str(mode)))
    
    if mode=='MA':  variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K'),"rb"))
    if mode=='ACA': variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K_ACA'),"rb"))
    if mode=='MA':  indicators = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','indicators'),"rb"))
    if mode=='ACA': indicators = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','indicators_ACA'),"rb"))
    if mode=='MA':  direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings'),'rb'))
    if mode=='ACA': direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings_ACA'),'rb'))
    if mode=='MA':  ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble'),"rb"))
    if mode=='ACA': ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble_ACA'),"rb"))

    df_member = _create_df(input_table)
    del input_table
    gc.collect()

    # check for direct mappings
    member_known = {}
    for idx, row in df_member.iterrows():
        knowns = []
        for i in row['CODE']:
            if i in direct_mappings and i not in knowns: knowns.append(direct_mappings[i])
        member_known[row['MEMBER']] = knowns
    
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
    df_vector = df_member[['MEMBER', 'CODE']]
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
            scores = v['classifier'].predict_proba(input_data)
            df_vector = pd.concat([df_vector, pd.DataFrame({k: scores[:,1]})], axis=1)
    del df_vector['temp']
    gc.collect()

    # modify knowns
    print("Modify Probabilities...")
    for index, value in df_member.iterrows():
        for col in member_known[value['MEMBER']]: df_vector.at[index, col] = 0.0
    df_vector = df_vector.dropna(axis=1, how='any')
    df_vector = df_vector.round({HCC: 4 for HCC in df_vector.columns if HCC.startswith('HCC')})

    # add values    
    # print("Calculating financial values...")
    # df_vector['PRED_VALUE'] = np.nan
    # for index, v in df_vector.iterrows():
    #     prob = []
    #     classifier_names = list(ensemble.keys())
    #     for classifier in classifier_names:
    #         if classifier in df_vector.columns: prob.append(df_vector.at[index, classifier])
    #         else: prob.append(0.0)
    #     prob, classifier_names = (list(t) for t in zip(*sorted(zip(prob ,classifier_names))))
    #     classifier_names = [values[i] for i in classifier_names]
    #     fin_value = round(sum(prob[i]*classifier_names[i] for i in range(len(prob))),2)
    #     df_vector.at[index, 'PRED_VALUE'] = fin_value

    if output_path:
        print("Saving results...")
        # df_vector = df_vector.sort_values('PRED_VALUE', ascending=False)
        df_vector.to_excel(output_path, sheet_name='output', header=True, index=False)
    
    df_condition = pd.melt(df_vector, id_vars=['MEMBER'], value_vars=[i for i in df_vector.columns if i.startswith('HCC')])
    df_condition.columns = ['MEMBER', 'HCC', 'CONFIDENCE']
    df_condition = df_condition.loc[df_condition['CONFIDENCE'] >= threshold, :]
    filtered_condtions = df_condition.groupby('MEMBER')['HCC'].agg(list).to_dict()
    
    if get_indicators:
        print("Finding indicators...")
        del df_vector
        gc.collect()
        df_member['FEATURE_IMPORTANCE'] = np.nan
        df_member['FEATURE_IMPORTANCE'] = df_member['FEATURE_IMPORTANCE'].astype(object)
        input_data = input_data.toarray().round(6)
        df_list = []
        for HCC in indicators.keys():
            df_HCC_member = df_member.copy(deep=True)
            df_HCC_member['HCC'] = HCC
            coef_matrix = input_data * indicators[HCC][0]
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
                df_HCC_member.at[idx, 'ENCOUNTER_ID'] = [df_member.at[idx, 'ENCOUNTER_ID'][i] for i in indices]
                df_HCC_member.at[idx, 'FEATURE_IMPORTANCE'] = list(coef_dict.values())
            df_list.append(df_HCC_member)
        df_member = pd.concat(df_list, axis=0)
        del input_data, df_list
        gc.collect()
#        df_CODE = df_member.set_index(['MEMBER', 'HCC'])['CODE'].apply(pd.Series).stack().reset_index().sort_values(['MEMBER', 'HCC', 'level_2'])
#        df_EIDS = df_member.set_index(['MEMBER', 'HCC'])['ENCOUNTER_ID'].apply(pd.Series).stack().reset_index().sort_values(['MEMBER', 'HCC', 'level_2'])[0]
#        df_COEF = df_member.set_index(['MEMBER', 'HCC'])['FEATURE_IMPORTANCE'].apply(pd.Series).stack().reset_index().sort_values(['MEMBER', 'HCC', 'level_2'])[0]
#        df_member = pd.concat([df_CODE, df_EIDS, df_COEF], axis=1).drop('level_2', axis=1)
#        del df_CODE, df_EIDS, df_COEF
#        gc.collect()
#        df_member.columns = ['MEMBER', 'HCC', 'CODE', 'ENCOUNTER_ID', 'FEATURE_IMPORTANCE']
        df_condition = pd.merge(df_condition, df_member, on=['MEMBER', 'HCC'], how='left')
    
    print('End  : ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return [tuple(x) for x in df_condition.values]

def plot(memberID,payer,server,apply_map=True,date_start=None,date_end=None,file_date_lmt=None,mode='MA'):
    """
    Plots the probability of the patient having each HCC
    @param memberID: memberID (e.g. 1120565)
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param apply_map: if true, drops value/probability of mapped HCCS to 1
    @param date_start: as 'YYYY-MM-DD' to get claims data from
    @param date_end: as 'YYYY-MM-DD' to get claims data to
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"

    Examples
    --------
    >>> from shakespeare import plot
    >>> plot(1120565, payer="CD_HealthFirst", server="CARABWDB03", apply_map=True)
    [Bar plot]
    """
    # initialize
    assert mode in ['MA', 'ACA'], print("AttributeError: mode must be either 'MA' or 'ACA', {} provided instead.".format(str(mode)))
    
    if mode=='MA':  descs = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC'),"rb"))
    if mode=='ACA': descs = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC_ACA'),"rb"))
    if mode=='MA':  direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings'),'rb'))
    if mode=='ACA': direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings_ACA'),'rb'))
    if mode=='MA':  ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble'),"rb"))
    if mode=='ACA': ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble_ACA'),"rb"))
    if mode=='MA':  variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K'),"rb"))
    if mode=='ACA': variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K_ACA'),"rb"))
    classifier_names = [i for i in ensemble.keys()]
    member_codes     = fetch_db.member_codes(memberID,payer=payer,server=server,date_start=date_start,date_end=date_end,file_date_lmt=file_date_lmt)
    member_vector    = vectorizers.build_member_input_vector(member_codes,variables) # note clientID is ignored for now
    
    # check for direct mappings
    member_mapped_conditions = []
    if apply_map:
        for i in member_codes:
            if i in direct_mappings and i not in member_mapped_conditions: member_mapped_conditions.append(direct_mappings[i])

    # check for all classifiers
    prob = []
    for classifier in classifier_names:
        if classifier in member_mapped_conditions: prob.append(0.0)
        elif ensemble[classifier]['classifier']: prob.append(ensemble[classifier]['classifier'].predict_proba(member_vector)[0,1])
        else: prob.append(0.0)
    prob,classifier_names = (list(t) for t in zip(*sorted(zip(prob,classifier_names))))
    
    # plot results
    spacing  = 3
    barwidth = 0.2
    plt.figure(figsize=(5,((np.sum(len(classifier_names))+1) + (2)*spacing)*barwidth))
    plt.barh(np.arange(len(classifier_names)),prob,align='center',alpha=0.5)
    plt.title('Maximum Confidences of Detection of each Classifier')
    plt.xlabel('Maximum Confidence of Detection')
    plt.xlim((0,1))
    plt.ylim((-1*barwidth*4,len(classifier_names)-barwidth))
    plt.yticks(np.arange(len(classifier_names)),[descs[i]+' ['+i+']' for i in classifier_names])
    plt.grid(linestyle='dashed',axis='x')
    plt.show()

def value(memberID,payer,server,apply_map=True,date_start=None,date_end=None,file_date_lmt=None,mode='MA'):
    """
    Financially values a member based on the probabilities of each HCC
    @param memberID: memberID (e.g. 1120565)
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param apply_map: if true, drops value/probability of mapped HCCS to 1
    @param date_start: as 'YYYY-MM-DD' to get claims data from
    @param date_end: as 'YYYY-MM-DD' to get claims data to
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"

    Return
    --------
    Member's estimated value (float)

    Examples
    --------
    >>> from shakespeare import value
    >>> value(1120565, payer="CD_HealthFirst", server="CARABWDB03", apply_map=True)
    3593.11
    """

    # initialize
    assert mode in ['MA', 'ACA'], print("AttributeError: mode must be either 'MA' or 'ACA', {} provided instead.".format(str(mode)))

    if mode=='MA':  values = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC_values'),'rb'))
    if mode=='ACA': values = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC_values_ACA'),'rb'))
    if mode=='MA':  direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings'),'rb'))
    if mode=='ACA': direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings_ACA'),'rb'))
    if mode=='MA':  ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble'),"rb"))
    if mode=='ACA': ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble_ACA'),"rb"))
    if mode=='MA':  variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K'),"rb"))
    if mode=='ACA': variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K_ACA'),"rb"))
    classifier_names = list(ensemble.keys())
    member_codes     = fetch_db.member_codes(memberID,payer=payer,server=server,date_start=date_start,date_end=date_end,file_date_lmt=file_date_lmt)
    member_vector    = vectorizers.build_member_input_vector(member_codes,variables) # note clientID is ignored for now
    
    # check for direct mappings
    member_mapped_conditions = []
    if apply_map:
        for i in member_codes:
            if i in direct_mappings and i not in member_mapped_conditions: member_mapped_conditions.append(direct_mappings[i])

    # check for all classifiers
    prob = []
    for classifier in classifier_names:
        if classifier in member_mapped_conditions: prob.append(0.0)
        elif ensemble[classifier]['classifier']: prob.append(ensemble[classifier]['classifier'].predict_proba(member_vector)[0,1])
        else: prob.append(0.0)
    prob,classifier_names = (list(t) for t in zip(*sorted(zip(prob,classifier_names))))
    classifier_names = [values[i] for i in classifier_names]
    return round(sum(prob[i]*classifier_names[i] for i in range(len(prob))),2)

def order_by_value(memberID_list,payer,server,apply_map=True,date_start=None,date_end=None,file_date_lmt=None,mode='MA'):
    """
    Given a list of members, orders the members in descending value to Inovalon.
    @param memberID_list: a list of memberIDs (e.g. [1120565,...])
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param apply_map: if true, drops value/probability of mapped HCCS to 1
    @param date_start: as 'YYYY-MM-DD' to get claims data from
    @param date_end: as 'YYYY-MM-DD' to get claims data to
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"

    Examples
    --------
    >>> from shakespeare import order_by_value
    >>> order_by_value([30352,1120565], payer="CD_HealthFirst", server="CARABWDB03", apply_map=True)
    Member: 30352    Est. Value: $6601.09
    Member: 1120565  Est. Value: $3593.11
    """
    values = sorted([(i,value(i,payer,server,apply_map,date_start,date_end,file_date_lmt,mode)) for i in memberID_list], key=lambda x: x[1],reverse=True)
    for i in values: print('Member: ' + str(i[0]) + '\t Est. Value: $' + str(i[1]))
    return

def support_evidence(memberID, payer='CD_HEALTHFIRST', server='CARABWDB03', date_start=None, date_end=None, file_date_lmt=None, mode='MA'):
    '''
    Get supporting evidences of coefficients to each HCC of a member
    @param memberID: one memberID (e.g. 1120565)
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located (e.g. 'CARABWDB03')
    @param date_start: as 'YYYY-MM-DD' to get claims data from
    @param date_end: as 'YYYY-MM-DD' to get claims data to
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"
    
    Return
    --------
    A dict of supporting evidences {k:HCC, v: supporting evidences {k:code, v:list of top n variable}}
    '''
    assert mode in ['MA', 'ACA'], print("AttributeError: mode must be either 'MA' or 'ACA', {} provided instead.".format(str(mode)))
    
    if mode=='MA':  HCCs = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC'),"rb"))
    if mode=='ACA': HCCs = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC_ACA'),"rb"))
    if mode=='MA':  ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble'),"rb"))
    if mode=='ACA': ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble_ACA'),"rb"))
    member_codes = fetch_db.member_codes(memberID, payer=payer, server=server, date_start=date_start, date_end=date_end, file_date_lmt=file_date_lmt)
    
    evidences = {HCC:{} for HCC in HCCs.keys()}
    for classifier in ensemble.keys():
        if not ensemble[classifier]['classifier']: continue
    
        if mode=='MA':  variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K'),"rb"))
        if mode=='ACA': variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K_ACA'),"rb"))
        if 'XGBClassifier' in str(ensemble[classifier]['classifier'].__str__): coefs = xgb_coef.coef(ensemble[classifier]['classifier'])
        if 'CalibratedClassifierCV' in str(ensemble[classifier]['classifier'].__str__):
            coefs = [xgb_coef.coef(c.base_estimator) for c in ensemble[classifier]['classifier'].calibrated_classifiers_]
            coefs = np.sum(coefs, axis=0) / ensemble[classifier]['classifier'].cv

        if isinstance(coefs,np.ndarray): coefs = coefs.tolist()
        else: coefs = list(coefs.toarray())
        assert len(coefs)==len(variables),"ERROR: variable-coefficient size mismatch. Aborting."
        
        # select top n
        combined = [i for i in zip(variables,coefs)]
        combined.sort(key=lambda x:x[1], reverse=True)
        variables = [i[0] for i in combined]
        coefs = [i[1] for i in combined]
        coefs,variables = (list(t) for t in zip(*sorted(zip(coefs,variables),reverse=True)))

        indices = [variables.index(i) for i in member_codes if i in variables]
        variables = [variables[i] for i in indices]
        coefs = [coefs[i] for i in indices]
        variables = variables[::-1]
        coefs = coefs[::-1]
        
        # format variables
        codes = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes'),"rb"))
        variables = [codes[i]+'  '+i if i in codes else i for i in variables]
        evidence = {v:coefs[i] for i,v in enumerate(variables)}
        evidences[classifier] = evidence
        
    return evidences

###########################  PRIVATE FUNCTIONS  ###############################

def _create_df(table):
    df = pd.DataFrame(table)
    if df.shape[1] == 4:
        df.columns = ['MEMBER','MEMBER_CLIENT_ID','ENCOUNTER_ID','CODE']
        df = df.groupby(['MEMBER','MEMBER_CLIENT_ID', 'CODE'])['ENCOUNTER_ID'].first().reset_index()
        df = df.groupby(['MEMBER','MEMBER_CLIENT_ID'])['CODE', 'ENCOUNTER_ID'].agg(list).reset_index()
    if df.shape[1] == 3:
        df.columns = ['MEMBER','ENCOUNTER_ID','CODE']
        df = df.groupby(['MEMBER', 'CODE'])['ENCOUNTER_ID'].first().reset_index()
        df = df.groupby(['MEMBER'])['CODE', 'ENCOUNTER_ID'].agg(list).reset_index()
    return df

def _top_percentile_variables(variables,coefs,n=0.04):
    """
    @param variables: a list of variables
    @param coefs: nparray or list of associated coefs
    @param n: if float, top decimal percent of vocab. int is top n
    """
    if isinstance(n,float): n = int(n*len(variables))
    if isinstance(coefs,np.ndarray): coefs = coefs.tolist()
    else: coefs = list(coefs.toarray())
    assert len(coefs)==len(variables),"ERROR: Vocab-coefficients size mismatch"

    # select top n
    combined = [i for i in zip(variables,coefs)]
    combined.sort(key=lambda x:x[1], reverse=True)
    vocab = [i[0] for i in combined]
    coefs = [i[1] for i in combined]
    coefs,vocab = (list(t) for t in zip(*sorted(zip(coefs,vocab),reverse=True)))
    vocab = vocab[0:n]
    return vocab
                    