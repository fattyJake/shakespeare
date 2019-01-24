# -*- coding: utf-8 -*-
###############################################################################
# Module:      intervention
# Description: repo of all functionalities for intervention
# Authors:     William Kinsman, Yage Wang
# Created:     06/18/2018
###############################################################################

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
import matplotlib.pyplot as plt
from . import fetch_db
from . import vectorizers
from . import xgb_coef

def detect(payer,server,memberID_list=None,D1=None,D2=None,D3=None,file_date_lmt=None,mem_date_start=None,mem_date_end=None,mode='MA',output_path=None,get_indicators=False):
    """
    Detects the HCCs a patient may have, and supporting evidence

    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param memberID_list: list of memberIDs (e.g. [1120565]); if None, get results from all members under payer
    @param apply_map: if true, sets value/probability of mapped HCCS to 1
    @param D1: as 'YYYY-MM-DD' to get claims data from
    @param D2: as 'YYYY-MM-DD' to get claims data to
    @param D3: as 'YYYY-MM-DD' to get claims data to from date_end for intervention purpose; if None, apply mapping to phase 1; else to phase 2
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"
    @param output_path: if not None, provided with the path of output EXCEL sheet to store HCC probs, financial values for all member list
    @param get_indicators: if False, only return probabilities for each HCC each member; if True, add HCC descriptions and supporting evidences of each member to each HCC
   
    Return
    --------
    member's condition:
        {k:memberID, v:{k:HCC (also predicted value and memberClientID), v:probablity of HCC}
        OR
        {k:memberID, v:{k:HCC (also predicted value and memberClientID), v:{confidence: float; description: str of HCC description; indications: list of codes along with their descriptions}}}

    Examples
    --------
    >>> from shakespeare.intervention import detect
    >>> detect(payer="CD_HealthFirst", server="CARABWDB03", [1120565], D1='2017-01-01', D2='2017-12-31', D3='2018-12-31', get_indicators=True)
        {1120565: {'MEMBER_CLIENT_ID': '130008347',
                   'HCC100': {'description': 'Ischemic or Unspecified Stroke',
                              'confidence': 0.003363140877909014,
                              'indications': [['CPT-3079F', 'DIAST BP 80-89 MM HG'],
                                              ['ICD10-M5126', 'Other intervertebral disc displacement, lumbar region'],
                                              ...
                                              ['CPT-86704', 'HEP B CORE ANTIBODY TOTAL'],
                                              ['CPT-86225', 'DNA ANTIBODY NATIVE']]},
         ...
                  'HCC114': {'description': 'Aspiration and Specified Bacterial Pneumonias',
                             'confidence': 0.00015744471629154034,
                             'indications': [['CPT-3079F', 'DIAST BP 80-89 MM HG'],
                                             ['ICD10-M5126', 'Other intervertebral disc displacement, lumbar region'],
                                             ...
                                             ['CPT-86704', 'HEP B CORE ANTIBODY TOTAL'],
                                             ['CPT-86225', 'DNA ANTIBODY NATIVE']]},
                  'PRED_VALUE': 8539.71}}
    """

    # initialize
    print('Start: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    assert mode in ['MA', 'ACA'], print("AttributeError: mode must be either 'MA' or 'ACA', {} provided instead.".format(str(mode)))
    
    if mode=='MA':  values = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC_values'),'rb'))
    if mode=='ACA': values = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_HCC_values_ACA'),'rb'))
    if mode=='MA':  variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K'),"rb"))
    if mode=='ACA': variables = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','variables_10K_ACA'),"rb"))
    if mode=='MA':  top_indicators = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','top_indicators'),"rb"))
    if mode=='ACA': top_indicators = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','top_indicators_ACA'),"rb"))
    if mode=='MA':  direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings'),'rb'))
    if mode=='ACA': direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings_ACA'),'rb'))
    if mode=='MA':  ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble'),"rb"))
    if mode=='ACA': ensemble = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','ensemble_ACA'),"rb"))
    HCC_exclusion = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','HCC_exclusion'),"rb"))

    print("Getting data of prior year...")
    table_1 = fetch_db.batch_member_codes(payer=payer, server=server, memberIDs=memberID_list, date_start=D1, date_end=D2, file_date_lmt=file_date_lmt, mem_date_start=mem_date_start, mem_date_end=mem_date_end)
    print("Getting data of service year...")
    table_2 = fetch_db.batch_member_codes(payer=payer, server=server, memberIDs=memberID_list, date_start=D2, date_end=D3, file_date_lmt=file_date_lmt, mem_date_start=mem_date_start, mem_date_end=mem_date_end)
    print("Finished...")
    if not table_1: return {}

    df_member = _create_df(table_1)
    if table_2: df_next_member = _create_df(table_2)
#    df_member = df_member[df_member['MEMBER'].isin(memberID_list)]
#    if table_2: df_next_member = df_next_member[df_next_member['MEMBER'].isin(memberID_list)]

    # check for direct mappings
    member_known      = {}
    next_member_known = {}
    for idx, row in df_member.iterrows():
        knowns = []
        for i in row['Code']:
            if i in direct_mappings and i not in knowns: knowns.append(direct_mappings[i])
        member_known[row['MEMBER']] = knowns
    
    if table_2: 
        for idx, row in df_next_member.iterrows():
            knowns = []
            for i in row['Code']:
                if i in direct_mappings and i not in knowns: knowns.append(direct_mappings[i])
            next_member_known[row['MEMBER']] = knowns
    
    # Don't miss members without any claims
    all_members      = dict(fetch_db.get_members(payer, server, mem_date_start, mem_date_end))
    all_members      = {i: all_members[i] for i in memberID_list}
    existing_members = list(df_member['MEMBER'])
    other_members    = [(i, all_members[i])for i in list(set(all_members)-set(existing_members))]
    if other_members:
        df_other         = pd.DataFrame(other_members)
        df_other.columns = ['MEMBER','MEMBER_CLIENT_ID']
        df_other['Code'] = [[] for _ in range(len(df_other))]
        df_member        = pd.concat([df_member, df_other], ignore_index=True, sort=False)
    for ID in all_members:
        if ID not in member_known:      member_known[ID] = []
        if ID not in next_member_known: next_member_known[ID] = []
    print("Total members: {}".format(str(df_member.shape[0])))

    cached_codes      = {row['MEMBER']:row['Code'] for _, row in df_member.iterrows()}
    if table_2:
        cached_next_codes = {row['MEMBER']:row['Code'] for _, row in df_next_member.iterrows()}
        del df_next_member
    
    # Vectorizing
    print("Vectoring...")
    df_member['temp'] = np.nan
    df_member['temp'] = df_member['temp'].astype(object)
    
    for idx, row in df_member.iterrows():
        vector = vectorizers.build_member_input_vector(row['Code'], variables)
        df_member.at[idx, 'temp'] = vector
    del df_member['Code']
    
    print("Running ML...")
    input_data = csr_matrix(vstack(df_member['temp'].tolist()))
    for k, v in ensemble.items():
        if v['classifier']:
            scores = v['classifier'].predict_proba(input_data)
            df_member = pd.concat([df_member, pd.DataFrame({k: scores[:,1]})], axis=1)
    del df_member['temp']

    # modify knowns
    print("Calculating financial values...")
    for index, value in df_member.iterrows():
        for col in member_known[value['MEMBER']]:
            if col not in HCC_exclusion[mode]: df_member.at[index, col] = 1.0
        for col in next_member_known[value['MEMBER']]: df_member.at[index, col] = 0.0

    # add values
    df_member = df_member.dropna(axis=1, how='any')
    df_member['PRED_VALUE'] = np.nan
    for index, v in df_member.iterrows():
        prob = []
        classifier_names = list(ensemble.keys())
        for classifier in classifier_names:
            if classifier in df_member.columns: prob.append(df_member.at[index, classifier])
            else: prob.append(0.0)
        prob, classifier_names = (list(t) for t in zip(*sorted(zip(prob ,classifier_names))))
        classifier_names = [values[i] for i in classifier_names]
        fin_value = round(sum(prob[i]*classifier_names[i] for i in range(len(prob))),2)
        df_member.at[index, 'PRED_VALUE'] = fin_value

    if output_path:
        print("Saving results...")
        df_member = df_member.sort_values('PRED_VALUE', ascending=False)
        df_member.to_excel(output_path, sheet_name='output', header=True, index=False)

    df_member = df_member.set_index('MEMBER')
    condition_dict = df_member.to_dict('index')

    if get_indicators:
        print("Finding indicators...")
        del df_member
        # create an organized output dictionary (Key=HCC, Values{'description' and 'indications' as (code,description)})
        for member, conditions in tqdm(condition_dict.items()):
            prob = conditions[k]
            for k in [i for i in list(conditions.keys()) if i.startswith('HCC')]:
                # get mapped codes and indicators. shortcircuit if mapped current year.
                if k in next_member_known[member]:
                    condition_indicators = [code for code,hcc in direct_mappings.items() if code in cached_next_codes[member] and k==hcc]
                    condition_dict[member][k] = {'confidence':prob,'indications':[[i,'CURRENTLY DIRECT MAPPING TO ' + direct_mappings[i]] for i in condition_indicators if i in cached_next_codes[member]]}
                    continue
                
                # get mapped codes and indicators. shortcircuit if mapped prior year.
                if k in member_known[member]:
                    condition_indicators = [code for code,hcc in direct_mappings.items() if code in cached_codes[member] and k==hcc]
                    condition_dict[member][k] = {'confidence':prob,'indications':[[i,'PREVIOUSLY DIRECT MAPPING TO ' + direct_mappings[i]] for i in condition_indicators if i in cached_codes[member]]}
                    continue

                # get probabilistic codes and indicators
                condition = {'confidence':prob, 'indications': [[i, top_indicators[k][i]] for i in cached_codes[member] if i in top_indicators[k]]}
                condition_dict[member][k] = condition
    print('End  : ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return condition_dict

def plot(memberID,payer,server,apply_map=True,D1=None,D2=None,D3=None,file_date_lmt=None,mode='MA'):
    """
    Plots the probability of the patient having each HCC
    @param memberID: memberID (e.g. 1120565)
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param apply_map: if true, drops value/probability of mapped HCCS to 1
    @param D1: as 'YYYY-MM-DD' to get claims data from
    @param D2: as 'YYYY-MM-DD' to get claims data to
    @param D3: as 'YYYY-MM-DD' to get claims data to from date_end for intervention purpose
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"

    Examples
    --------
    >>> from shakespeare.intervention import plot
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
    HCC_exclusion = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','HCC_exclusion'),"rb"))
    classifier_names = [i for i in ensemble.keys()]
    member_codes      = fetch_db.member_codes(memberID,payer=payer,server=server,date_start=D1,date_end=D2,file_date_lmt=file_date_lmt)
    next_member_codes = fetch_db.member_codes(memberID,payer=payer,server=server,date_start=D2,date_end=D3)
    member_vector     = vectorizers.build_member_input_vector(member_codes,variables) # note clientID is ignored for now
    
    # check for direct mappings
    member_mapped_conditions = []
    next_member_mapped_conditions = []
    if apply_map:
        for i in member_codes:
            if i in direct_mappings and i not in member_mapped_conditions: member_mapped_conditions.append(direct_mappings[i])
        for i in next_member_codes:
            if i in direct_mappings and i not in next_member_mapped_conditions: next_member_mapped_conditions.append(direct_mappings[i])
    
    # check for all classifiers
    prob = []
    for classifier in classifier_names:
        if classifier in member_mapped_conditions and classifier not in HCC_exclusion[mode]: prob.append(1.0)
        elif classifier in next_member_mapped_conditions: prob.append(0.0)
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

def value(memberID,payer,server,apply_map=True,D1=None,D2=None,D3=None,file_date_lmt=None,mode='MA'):
    """
    Financially values a member based on the probabilities of each HCC
    @param memberID: memberID (e.g. 1120565)
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param apply_map: if true, drops value of mapped HCCS to 0
    @param D1: as 'YYYY-MM-DD' to get claims data from
    @param D2: as 'YYYY-MM-DD' to get claims data to
    @param D3: as 'YYYY-MM-DD' to get claims data to from date_end for intervention purpose
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"

    Return
    --------
    Member's estimated value (float)

    Examples
    --------
    >>> from shakespeare.intervention import value
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
    HCC_exclusion = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','HCC_exclusion'),"rb"))
    classifier_names = list(ensemble.keys())
    member_codes      = fetch_db.member_codes(memberID,payer=payer,server=server,date_start=D1,date_end=D2)
    next_member_codes = fetch_db.member_codes(memberID,payer=payer,server=server,date_start=D2,date_end=D3)
    member_vector     = vectorizers.build_member_input_vector(member_codes,variables) # note clientID is ignored for now
    
    # check for direct mappings
    member_mapped_conditions = []
    next_member_mapped_conditions = []
    if apply_map:
        for i in member_codes:
            if i in direct_mappings and i not in member_mapped_conditions: member_mapped_conditions.append(direct_mappings[i])
        for i in next_member_codes:
            if i in direct_mappings and i not in next_member_mapped_conditions: next_member_mapped_conditions.append(direct_mappings[i])
    
    # check for all classifiers
    prob = []
    for classifier in classifier_names:
        if classifier in member_mapped_conditions and classifier not in HCC_exclusion[mode]: prob.append(1.0)
        elif classifier in next_member_mapped_conditions: prob.append(0.0)
        elif ensemble[classifier]['classifier']: prob.append(ensemble[classifier]['classifier'].predict_proba(member_vector)[0,1])
        else: prob.append(0.0)
    prob,classifier_names = (list(t) for t in zip(*sorted(zip(prob,classifier_names))))
    classifier_names = [values[i] for i in classifier_names]
    return round(sum(prob[i]*classifier_names[i] for i in range(len(prob))),2)

def order_by_value(memberID_list,payer,server,apply_map=True,D1=None,D2=None,D3=None,file_date_lmt=None,mode='MA'):
    """
    Given a list of members, orders the members in descending value to Inovalon.
    @param memberID_list: a list of memberIDs (e.g. [1120565,...])
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param apply_map: if true, drops value of mapped HCCS to 0
    @param D1: as 'YYYY-MM-DD' to get claims data from
    @param D2: as 'YYYY-MM-DD' to get claims data to
    @param D3: as 'YYYY-MM-DD' to get claims data to from date_end for intervention purpose
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"

    Examples
    --------
    >>> from shakespeare.intervention import order_by_value
    >>> order_by_value([30352,1120565], payer="CD_HealthFirst", server="CARABWDB03", apply_map=True)
    Member: 30352    Est. Value: $6601.09
    Member: 1120565  Est. Value: $3593.11
    """
    values = sorted([(i,value(i,payer,server,apply_map,D1,D2,D3,file_date_lmt,mode)) for i in memberID_list], key=lambda x: x[1],reverse=True)
    for i in values: print('Member: ' + str(i[0]) + '\t Est. Value: $' + str(i[1]))
    return

def support_evidence(memberID, payer='CD_HEALTHFIRST', server='CARABWDB03', D1=None, D2=None, file_date_lmt=None, mode='MA'):
    '''
    Get supporting evidences of coefficients to each HCC of a member
    @param memberID: one memberID (e.g. 1120565)
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located (e.g. 'CARABWDB03')
    @param D1: as 'YYYY-MM-DD' to get claims data from
    @param D2: as 'YYYY-MM-DD' to get claims data to
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
    mem_codes = fetch_db.member_codes(memberID, payer=payer, server=server, date_start=D1, date_end=D2, file_date_lmt=file_date_lmt)

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

        indices = [variables.index(i) for i in mem_codes if i in variables]
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
    df.columns = ['MEMBER','MEMBER_CLIENT_ID','Code']
    df = pd.DataFrame(df.groupby(['MEMBER','MEMBER_CLIENT_ID'])['Code'].agg(list))
    df['index'] = df.index
    df[['MEMBER','MEMBER_CLIENT_ID']] = df['index'].apply(pd.Series)
    del df['index']
    df = df.reset_index(drop=True)
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
