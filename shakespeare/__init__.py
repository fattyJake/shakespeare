# -*- coding: utf-8 -*-
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
# A python package designed to detect gaps using client claim data
#
# Authors:  William Kinsman, Chenyu (Oliver) Ha, Yage Wang
# Created:  11.02.2017
# Version:  2.6.0
###############################################################################


# system libs
import os
import pickle
import itertools
import operator
import gc
from datetime import datetime

# data & machine learning libs
import numpy as np
import pandas as pd
import shap
from scipy.sparse import csr_matrix, vstack

# package libs
from . import fetch_db
from . import training
from . import visualizations
from . import utils


def detect_internal(
    payer: str,
    server: str,
    date_start: str,
    date_end: str,
    memberID_list: list = None,
    file_date_lmt: str = None,
    mem_date_start: str = None,
    mem_date_end: str = None,
    model: int = 63,
    auto_update: bool = False,
    threshold: float = 0,
    get_indicators: bool = False,
    top_n_indicator: int = 5,
):
    """
    Wrapper of core_ml for Inovalon internal use.

    Parameters
    --------
    payer : str
        name of payer table (e.g. 'CD_HEALTHFIRST')
        
    server : str
        CARA server on which the payer is located ('CARABWDB03')
        
    date_start : str
        string as 'YYYY-MM-DD' to get claims data from
        
    date_end : str
        string as 'YYYY-MM-DD' to get claims data to

    memberID_list : list, optional (default: None)
        list of memberIDs (e.g. [1120565]); if None, get results from all
        members under payer

    file_date_lmt : str, optional (default: None)
        string as 'YYYY-MM-DD' indicating the latest file date limit of patient
        codes, generally at the half of one year
     
    mem_date_start : str, optional (default: None)
        as 'YYYY-MM-DD', starting date to filter memberIDs by dateInserted
        
    mem_date_end : str, optional (default: None)
        as 'YYYY-MM-DD', ending date to filter memberIDs by dateInserted
        
    model : int, optional (default: 63)
        an integer of model version ID in
        MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only
        support 'CDPS', 'CMS' and 'HHS'
        
    auto_update : boolean, optional (default: False)
        if True, and no matching model pickled, it will call `update` function
        first
        
    threshold : float, optional (default: 0)
        a float between 0 and 1 for filtering output confidence above it
        
    get_indicators : boolean, optional (default: False)
        if False, only return probabilities for each HCC each member; if True,
        add supporting evidences of each member to each HCC
        
    top_n_indicator : int, optional (default: 5)
        how many indicators to output for each member each HCC
    
    Return
    --------
    member's condition:
        [(memberID, HCC, prob, tp_flag)]
        OR
        [(memberID, HCC, prob, tp_flag, top_n_codes, top_n_encounter_id,
            top_n_coefficient)]

    Examples
    --------
    >>> from shakespeare import detect_internal
    >>> detect_internal(
            payer="CD_GATEWAY",
            server="CARABWDB06",
            date_start='2017-01-01',
            date_end='2017-12-31',
            threshold=0.9,
            top_n_indicator=5,
            get_indicators=True
        )

        [(33330, 'HCC19', 0.9041, 0.9800, 0,
          ['HCPCS-A5500', 'CPT-82043', 'ICD10-I10', 'CPT-83036', 'ICD10-E785'],
          [167816,174530,219484],
          [1.4374, 0.7924, 0.4818, 0.38, 0.2694]),
         (33333, 'HCC19', 0.8445, 0.9668,, 0,
          ['NDC9-000882219', 'ICD10-I10', 'CPT-82043', 'CPT-83036', 'ICD10-E039'],
          [11199,164276,152785],
          [1.3392, 0.6136, 0.5779, 0.4449, 0.3895]),
         ...
         (111382, 'HCC114', 0.6686, 0.9226, 1,
          ['ICD10-J189', 'ICD10Proc-5A1955Z', 'ICD10-R918', 'ICD10-Z4682', 'ICD10-Z66'],
          [123838, 164643, 176622, 76237, 83652],
          [1.987, 1.0033, 0.6202, 0.5484, 0.4784]),
         (312068, 'HCC114', 0.6505, 0.9176, 1,
          ['ICD10-J189', 'ICD10Proc-5A1955Z', 'ICD10-Z4682', 'CPT-99233', 'ICD10-R918'],
          [114646, 118574, 118237, 24822, 149849, 14239],
          [1.8562, 1.5598, 0.8454, 0.6191, 0.6101])]
    """

    # initialize
    if memberID_list and not isinstance(memberID_list, list):
        memberID_list = [memberID_list]
    if not memberID_list:
        memberID_list = list(
            {
                tup[0]
                for tup in fetch_db.get_members(
                    payer, server, mem_date_start, mem_date_end
                )
            }
        )
    memberID_list = list(set(memberID_list))

    if date_end:
        year = int(date_end[:4])
    else:
        year = str(datetime.now().year)

    if not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            r"pickle_files",
            r"ensembles",
            f"ensemble_{model}",
        )
    ):
        if auto_update:
            print("Model version not exist yet! Updating...")
            update(model, year)
        else:
            raise FileNotFoundError(
                f"Model version not exist! Please use `shakespeare."
                f"update({model}, {year})` to train ML for Model {model}."
            )

    print("Getting data...")
    if len(memberID_list) <= 50000:
        table = fetch_db.batch_member_codes(
            payer=payer,
            server=server,
            memberID_list=memberID_list,
            date_start=date_start,
            date_end=date_end,
            file_date_lmt=file_date_lmt,
            mem_date_start=mem_date_start,
            mem_date_end=mem_date_end,
            model=model,
        )

        if table.shape[0] == 0:
            return []
        return core_ml(
            table,
            year,
            model,
            auto_update,
            threshold,
            get_indicators,
            top_n_indicator,
        )
    else:
        memberID_list = [
            memberID_list[i : i + 40000]
            for i in range(0, len(memberID_list), 40000)
        ]
        # TODO: change results structure once output fixed
        results = []
        print(f"Total batches: {len(memberID_list)}")
        for batch, sub_mem in enumerate(memberID_list):
            print(
                f"###################### Processing batch {batch+1} "
                + "######################"
            )
            table = fetch_db.batch_member_codes(
                payer=payer,
                server=server,
                memberID_list=sub_mem,
                date_start=date_start,
                date_end=date_end,
                file_date_lmt=file_date_lmt,
                mem_date_start=mem_date_start,
                mem_date_end=mem_date_end,
                model=model,
            )

            if table.shape[0] == 0:
                continue
            results.extend(
                core_ml(
                    table,
                    year,
                    model,
                    auto_update,
                    threshold,
                    get_indicators,
                    top_n_indicator,
                )
            )
            del table
            gc.collect()

        return results


def detect_api(json_body: dict):
    """
    Wrapper of core_ml for external API use.

    Parameters
    --------
    json_body : dict
        the API body validated by Flask application
    
    Return
    --------
    member's condition:
        [(memberID, HCC, prob, norm_prob, tp_flag)]
        OR
        [(memberID, HCC, prob, norm_prob, tp_flag, top_n_codes,
            top_n_encounter_id, top_n_coefficient)]

    Examples
    --------
    >>> from shakespeare import detect_api
    >>> detect_api(payload)

        [(33330, 'HCC19', 0.9041, 0.9800, 0,
          ['HCPCS-A5500', 'CPT-82043', 'ICD10-I10', 'CPT-83036', 'ICD10-E785'],
          [167816,174530,219484],
          [1.4374, 0.7924, 0.4818, 0.38, 0.2694]),
         (33333, 'HCC19', 0.8445, 0.9668,, 0,
          ['NDC9-000882219', 'ICD10-I10', 'CPT-82043', 'CPT-83036', 'ICD10-E039'],
          [11199,164276,152785],
          [1.3392, 0.6136, 0.5779, 0.4449, 0.3895]),
         ...
         (111382, 'HCC114', 0.6686, 0.9226, 1,
          ['ICD10-J189', 'ICD10Proc-5A1955Z', 'ICD10-R918', 'ICD10-Z4682', 'ICD10-Z66'],
          [123838, 164643, 176622, 76237, 83652],
          [1.987, 1.0033, 0.6202, 0.5484, 0.4784]),
         (312068, 'HCC114', 0.6505, 0.9176, 1,
          ['ICD10-J189', 'ICD10Proc-5A1955Z', 'ICD10-Z4682', 'CPT-99233', 'ICD10-R918'],
          [114646, 118574, 118237, 24822, 149849, 14239],
          [1.8562, 1.5598, 0.8454, 0.6191, 0.6101])]
    """

    table = []
    for member_codes in json_body["payload"]:
        sub_table = pd.DataFrame(member_codes["codes"])
        sub_table["mem_id"] = member_codes["mem_id"]
        table.append(sub_table)

    table = pd.concat(table, axis=0)
    table["year"] = table["service_date"].str.slice(0, 4)
    table["year"] = table["year"].astype(int)
    table["code"] = table.apply(
        lambda row: f"{row['code_type']}-{row['code']}", axis=1
    )
    table = table[["mem_id", "pra_id", "spec_id", "year", "code"]]

    return core_ml(
        table,
        target_year=json_body.get("target_year", datetime.today().year),
        model=json_body["model_version_ID"],
        auto_update=False,  # TODO: allow auto_update in future?
        threshold=json_body.get("threshold", 0.0),
        get_indicators=json_body.get("get_indicators", True),
        top_n_indicator=json_body.get("top_n_indicator", 5),
    )


# TODO: determine parameter list
def core_ml(
    table: pd.DataFrame,
    target_year: int,
    model: int = 63,
    auto_update: bool = False,
    threshold: float = 0,
    get_indicators: bool = False,
    top_n_indicator: int = 5,
):
    """
    Detects the HCCs a patient may have, and supporting evidence

    Parameters
    --------
    table : pandas.DataFrame
        a table with coulumn ['mem_id', 'pra_id', 'spec_id', 'year', 'code']
    
    target_year : int
        target service year

    model : int, optional (default: 63)
        an integer of model version ID in
        MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only
        support 'CDPS', 'CMS' and 'HHS'
        
    auto_update : boolean, optional (default: False)
        if True, and no matching model pickled, it will call `update` function
        first
        
    threshold : float, optional (default: 0)
        a float between 0 and 1 for filtering output confidence above it
        
    get_indicators : boolean, optional (default: False)
        if False, only return probabilities for each HCC each member; if True,
        add supporting evidences of each member to each HCC
        
    top_n_indicator : int, optional (default: 5)
        how many indicators to output for each member each HCC
    
    Return
    --------
    member's condition:
        [(memberID, HCC, prob, norm_prob, tp_flag)]
        OR
        [(memberID, HCC, prob, norm_prob, tp_flag, top_n_codes,
            top_n_encounter_id, top_n_coefficient)]

    Examples
    --------
    >>> from shakespeare import core_ml
    >>> core_ml(
            table,
            2019,
            threshold=0.9,
            top_n_indicator=5,
            get_indicators=True
        )

        [(33330, 'HCC19', 0.9041, 0.9800, 0,
          ['HCPCS-A5500', 'CPT-82043', 'ICD10-I10', 'CPT-83036', 'ICD10-E785'],
          [167816,174530,219484],
          [1.4374, 0.7924, 0.4818, 0.38, 0.2694]),
         (33333, 'HCC19', 0.8445, 0.9668,, 0,
          ['NDC9-000882219', 'ICD10-I10', 'CPT-82043', 'CPT-83036', 'ICD10-E039'],
          [11199,164276,152785],
          [1.3392, 0.6136, 0.5779, 0.4449, 0.3895]),
         ...
         (111382, 'HCC114', 0.6686, 0.9226, 1,
          ['ICD10-J189', 'ICD10Proc-5A1955Z', 'ICD10-R918', 'ICD10-Z4682', 'ICD10-Z66'],
          [123838, 164643, 176622, 76237, 83652],
          [1.987, 1.0033, 0.6202, 0.5484, 0.4784]),
         (312068, 'HCC114', 0.6505, 0.9176, 1,
          ['ICD10-J189', 'ICD10Proc-5A1955Z', 'ICD10-Z4682', 'CPT-99233', 'ICD10-R918'],
          [114646, 118574, 118237, 24822, 149849, 14239],
          [1.8562, 1.5598, 0.8454, 0.6191, 0.6101])]
    """
    print("Start: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            r"pickle_files",
            r"ensembles",
            f"ensemble_{model}",
        )
    ):
        if auto_update:
            print("Model version not exist !")
            update(model, datetime.now().year)
        else:
            raise ValueError("Model version not exist !")

    variables = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"variables",
                f"variables_{model}",
            ),
            "rb",
        )
    )
    mappings = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"mappings",
                f"mapping_{model}",
            ),
            "rb",
        )
    )
    ensemble = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"ensembles",
                f"ensemble_{model}",
            ),
            "rb",
        )
    )

    MEMBER_LIST = list(table.mem_id.unique())
    print(f"Total members with codes: {len(MEMBER_LIST)}")
    dict_prior, dict_current = utils.preprocess_table(table, target_year)
    del table
    gc.collect()

    # check for known conditions
    # NOTE: member_known might not contain all members
    member_known_prior = {
        mem_id: set(
            itertools.chain(
                *[mappings.get(code, []) for code in record["code"]]
            )
        )
        for mem_id, record in dict_prior.items()
    }
    member_known_current = {
        mem_id: set(
            itertools.chain(
                *[mappings.get(code, []) for code in record["code"]]
            )
        )
        for mem_id, record in dict_current.items()
    }

    # Vectorizing
    print("Vectoring...")
    vector_prior = csr_matrix(
        vstack(
            [
                utils.build_member_input_vector(
                    dict_prior.get(mem_id, {'code': []})['code'], variables
                )
                for mem_id in MEMBER_LIST
            ]
        )
    )
    vector_current = csr_matrix(
        vstack(
            [
                utils.build_member_input_vector(
                    dict_prior.get(mem_id, {'code': []})['code']
                    + dict_current.get(mem_id, {'code': []})['code'],
                    variables,
                )
                for mem_id in MEMBER_LIST
            ]
        )
    )

    # running machine learning
    print("Running ML for retrospective analysis...")
    condition_retro = utils.run_ml(
        ensemble, MEMBER_LIST, vector_prior, threshold
    )
    condition_retro["known"] = condition_retro.apply(
        lambda x: 1 if x.hcc in member_known_prior.get(x.mem_id, []) else 0,
        axis=1,
    )

    print("Running ML for prospective anlysis...")
    condition_prosp = utils.run_ml(
        ensemble, MEMBER_LIST, vector_current, threshold
    )
    condition_prosp["known"] = condition_prosp.apply(
        lambda x: 1 if x.hcc in member_known_current.get(x.mem_id, []) else 0,
        axis=1,
    )
    # TODO: come up with better flag system of UCCC for pra_id matching purpose
    condition_prosp["uccc"] = condition_prosp.apply(
        lambda x: 1
        if x.hcc in member_known_prior.get(x.mem_id, []) and x.known == 0
        else 0,
        axis=1,
    )

    # TODO: engineer this process: optimize implementation; design output
    if get_indicators:
        print("Finding indicators...")

        filtered_condtions = (
            df_condition.groupby("mem_id")["hcc"].agg(list).to_dict()
        )

        df_list = []
        for HCC in ensemble.keys():
            if ensemble[HCC]["classifier"] is None:
                continue

            explainer = shap.TreeExplainer(
                ensemble[HCC]["classifier"]
                .calibrated_classifiers_[0]
                .base_estimator
            )
            df_HCC_member = df_member.copy(deep=True)
            df_HCC_member["HCC"] = HCC
            coef_matrix = explainer.shap_values(vector_prior)
            coef_matrix = vector_prior.toarray() * coef_matrix

            for idx, member in df_HCC_member.iterrows():
                if HCC not in filtered_condtions.get(member["MEMBER"], []):
                    continue

                codes = member["CODE"]
                i = list(np.where(coef_matrix[idx] > 0)[0])
                coef_dict = dict(
                    zip(
                        [variables[index] for index in i],
                        list(coef_matrix[idx][i]),
                    )
                )
                coef_dict = dict(
                    sorted(
                        coef_dict.items(),
                        key=operator.itemgetter(1),
                        reverse=True,
                    )[:top_n_indicator]
                )

                indices = [codes.index(code) for code in coef_dict]
                df_HCC_member.at[idx, "CODE"] = [codes[i] for i in indices]
                pra_list = itertools.chain(
                    *[df_member.at[idx, "PRA_ID"][i] for i in indices]
                )
                df_HCC_member.at[idx, "PRA_ID"] = list(
                    utils.unique_keeping_order(pra_list)
                )
                df_HCC_member.at[idx, "FEATURE_IMPORTANCE"] = list(
                    coef_dict.values()
                )
            df_list.append(df_HCC_member)
            del explainer, coef_matrix
            gc.collect()

        df_member = pd.concat(df_list, axis=0)
        del vector_prior, df_list
        gc.collect()
        df_condition = pd.merge(
            df_condition, df_member, on=["MEMBER", "HCC"], how="left"
        )

    print("End  : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return [tuple(x) for x in df_condition.values]


def update(model, year_of_service):
    """
    Since CARA use different ICD-HCC mappings at different service year, this
    function is for updating mappings, variable spaces and XGBoost models for
    all lines of business. 

    Parameters
    --------
    model : int
        an integer of model version ID in
        MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only
        support 'CDPS', 'CMS' and 'HHS'
        
    year_of_service : int
        a intiger of the year of service for training data retrieval purpose

    Examples
    --------
    >>> from shakespeare import update
    >>> update(63, 2018)
    ########################## Training New Model 63 ##########################
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
    
    ######################## Finished Training Model 63 #######################
    """
    assert isinstance(model, int) and year_of_service > 2010, print(
        "The year of service must be a integer greater than 2010"
    )

    print(
        "############################## Training New Model "
        + str(model)
        + " ##############################"
    )
    training_set = training.get_training_set(year_of_service, model)
    training.update_mappings(model)
    training.update_variables(training_set, model)
    training.update_ensembles(training_set, model)
    print(
        "############################ Finished Training Model "
        + str(model)
        + " ###########################"
    )


def delete(model):
    """
    Module for deleting old models to free up disk space

    Parameters
    --------
    model : int
        an integer of model version ID in
        MPBWDB1.CARA2_Controller.dbo.ModelVersions; note this package only
        support 'CDPS', 'CMS' and 'HHS'
    
    Examples
    --------
    >>> from shakespeare import delete
    >>> delete(63)
    """
    if os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            r"pickle_files",
            r"ensembles",
            "ensemble_{}".format(model),
        )
    ):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"ensembles",
                "ensemble_{}".format(model),
            )
        )
    if os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            r"pickle_files",
            r"mappings",
            "mapping_{}".format(model),
        )
    ):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"mappings",
                "mapping_{}".format(model),
            )
        )
    if os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            r"pickle_files",
            r"variables",
            "variables_{}".format(model),
        )
    ):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"variables",
                "variables_{}".format(model),
            )
        )
