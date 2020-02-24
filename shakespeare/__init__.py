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
# Authors:  Yage Wang, William Kinsman, Chenyu (Oliver) Ha
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
import pandas as pd
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
    mode: str = 'b',
    auto_update: bool = False,
    threshold: float = 0,
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

    mode : str
        one of 'r' for retrospective only, 'p' for prospective only and 'b' for
        both.

    auto_update : boolean, optional (default: False)
        if True, and no matching model pickled, it will call `update` function
        first

    threshold : float, optional (default: 0)
        a float between 0 and 1 for filtering output confidence above it

    top_n_indicator : int, optional (default: 5)
        how many indicators to output for each member each HCC. If 0, it won't
        return any supporting evidences

    Return
    --------
    final_results : dict
    {
        'retrospective': [
            {
                'mem_id': int,
                'gaps': [
                    {
                        'condition_category': str,
                        'confidence': float,
                        'known': bool,
                        *['top_indicators': list,]
                        *['provider_id': list]
                    }
                ]
            }
        ],
        'prospective': [
            {
                'mem_id': int,
                'gaps': [
                    {
                        'condition_category': str,
                        'confidence': float,
                        'known': bool,
                        'uccc': bool,
                        *['top_indicators': list,]
                        *['provider_id': list]
                    }
                ]
            }
        ]
    }

    Examples
    --------
    >>> from shakespeare import detect_internal
    >>> detect_internal(
            payer="CD_GATEWAY",
            server="CARABWDB06",
            date_start='2018-01-01',
            date_end='2019-04-15',
            threshold=0.9,
            top_n_indicator=5,
        )

        {
            "retrospective": [
                {
                    "mem_id": 23023,
                    "gaps": [
                        {
                            "condition_category": "HCC188",
                            "confidence": 0.990729,
                            "known": 1,
                            "top_indicators": [
                                "ICD10DX-K9423",
                                "ICD10DX-K9429",
                                "ICD10DX-Z931"
                            ],
                            "provider_id": [
                                505,
                                131157
                            ]
                        },
                        ...
                    ]
                },
                ...
            ],
            "prospective": [
                {
                    "mem_id": 23023,
                    "gaps": [
                        {
                            "condition_category": "HCC188",
                            "confidence": 0.98871,
                            "known": 0,
                            "uccc": 1,
                            "top_indicators": [
                                "ICD10DX-K9423",
                                "ICD10DX-K9429",
                                "ICD10DX-Z931"
                            ],
                            "provider_id": [
                                505,
                                131157
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """

    # initialize
    assert mode in ['r', 'p', 'b'], (
        'ValueError: "mode" can only be one of "r", "p" or "b", '
        f'got {mode} instead.'
    )

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

        table["year"] = table.service_date.map(lambda x: x.year)
        del table["service_date"]
        return core_ml(
            table,
            target_year=year,
            model=model,
            mode=mode,
            auto_update=auto_update,
            threshold=threshold,
            top_n_indicator=top_n_indicator,
        )
    else:
        memberID_list = [
            memberID_list[i: i + 40000]
            for i in range(0, len(memberID_list), 40000)
        ]

        results = {"retrospective": [], "prospective": []}
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

            table["year"] = table.service_date.map(lambda x: x.year)
            del table["service_date"]
            sub_results = core_ml(
                table,
                target_year=year,
                model=model,
                mode=mode,
                auto_update=auto_update,
                threshold=threshold,
                top_n_indicator=top_n_indicator,
            )
            results["retrospective"].extend(sub_results["retrospective"])
            results["prospective"].extend(sub_results["prospective"])

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
    final_results : dict
    {
        'retrospective': [
            {
                'mem_id': int,
                'gaps': [
                    {
                        'condition_category': str,
                        'confidence': float,
                        'known': bool,
                        *['top_indicators': list,]
                        *['provider_id': list]
                    }
                ]
            }
        ],
        'prospective': [
            {
                'mem_id': int,
                'gaps': [
                    {
                        'condition_category': str,
                        'confidence': float,
                        'known': bool,
                        'uccc': bool,
                        *['top_indicators': list,]
                        *['provider_id': list]
                    }
                ]
            }
        ]
    }

    Examples
    --------
    >>> from shakespeare import detect_api
    >>> detect_api(payload)

        {
            "retrospective": [
                {
                    "mem_id": 23023,
                    "gaps": [
                        {
                            "condition_category": "HCC188",
                            "confidence": 0.990729,
                            "known": 1,
                            "top_indicators": [
                                "ICD10DX-K9423",
                                "ICD10DX-K9429",
                                "ICD10DX-Z931"
                            ],
                            "provider_id": [
                                505,
                                131157
                            ]
                        },
                        ...
                    ]
                },
                ...
            ],
            "prospective": [
                {
                    "mem_id": 23023,
                    "gaps": [
                        {
                            "condition_category": "HCC188",
                            "confidence": 0.98871,
                            "known": 0,
                            "uccc": 1,
                            "top_indicators": [
                                "ICD10DX-K9423",
                                "ICD10DX-K9429",
                                "ICD10DX-Z931"
                            ],
                            "provider_id": [
                                505,
                                131157
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
    """

    table = []
    for member_codes in json_body["payload"]:
        sub_table = pd.DataFrame(member_codes["codes"])
        sub_table["mem_id"] = member_codes["mem_id"]
        table.append(sub_table)

    table = pd.concat(table, axis=0)

    # check code_type
    unique_code_type = list(table.code_type.unique())
    non_recognized_code_type = [
        ct
        for ct in unique_code_type
        if ct
        not in [
            "ICD9DX",
            "ICD10DX",
            "CPT",
            "HCPCS",
            "NDC9",
            "ICD9PX",
            "ICD10PX",
            "DRG",
        ]
    ]

    # check target year
    if "target_year" in json_body:
        no_target_year = False
        target_year = json_body["target_year"]
    else:
        no_target_year = True
        target_year = datetime.today().year

    table["year"] = table["service_date"].str.slice(0, 4)
    table["year"] = table["year"].astype(int)
    unique_service_years = list(table.year.unique())
    non_recognized_year = [
        str(y) for y in unique_service_years if y > target_year
    ]

    table["code"] = table.apply(
        lambda row: f"{row['code_type']}-{row['code']}", axis=1
    )
    table = table[["mem_id", "provider_id", "spec_id", "year", "code"]]

    final_results = core_ml(
        table,
        target_year=target_year,
        model=json_body["model_version_ID"],
        mode=json_body.get("mode", "b"),
        auto_update=False,
        threshold=json_body.get("threshold", 0.0),
        top_n_indicator=json_body.get("top_n_indicator", 5),
    )

    final_results["warnings"] = []
    if non_recognized_code_type:
        final_results["warnings"].append(
            "Detected code types that are not recognized: "
            + f"{', '.join(non_recognized_code_type)}. The model only accept "
            + 'code types "ICD9DX", "ICD10DX", "ICD9PX", "ICD10X", "CPT", "HCP'
            + 'CS", "NDC9" and "DRG". Please check and consider another try.'
        )
    if no_target_year:
        final_results["warnings"].append(
            "No target year provided, use today's year as target year instead."
        )
    if non_recognized_year:
        final_results["warnings"].append(
            "There are claims documented in service years that are later then "
            + f"target year: {', '.join(non_recognized_code_type)}. Please "
            + "check and consider another try."
        )

    return final_results


def core_ml(
    table: pd.DataFrame,
    target_year: int,
    model: int = 63,
    mode: str = 'b',
    auto_update: bool = False,
    threshold: float = 0,
    top_n_indicator: int = 5,
):
    """
    Detects the HCCs a patient may have, and supporting evidence

    Parameters
    --------
    table : pandas.DataFrame
        a table with coulumn ['mem_id', 'provider_id', 'spec_id', 'year', 'code']

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

    top_n_indicator : int, optional (default: 5)
        how many indicators to output for each member each HCC; If 0, it won't
        return any supporting evidences

    Return
    --------
    final_results : dict
    {
        'retrospective': [
            {
                'mem_id': int,
                'gaps': [
                    {
                        'condition_category': str,
                        'confidence': float,
                        'known': bool,
                        *['top_indicators': list,]
                        *['provider_id': list]
                    }
                ]
            }
        ],
        'prospective': [
            {
                'mem_id': int,
                'gaps': [
                    {
                        'condition_category': str,
                        'confidence': float,
                        'known': bool,
                        'uccc': bool,
                        *['top_indicators': list,]
                        *['provider_id': list]
                    }
                ]
            }
        ]
    }

    Examples
    --------
    >>> from shakespeare import core_ml
    >>> core_ml(
            table,
            2019,
            threshold=0.9,
            top_n_indicator=5,
        )

        {
            "retrospective": [
                {
                    "mem_id": 23023,
                    "gaps": [
                        {
                            "condition_category": "HCC188",
                            "confidence": 0.990729,
                            "known": 1,
                            "top_indicators": [
                                "ICD10DX-K9423",
                                "ICD10DX-K9429",
                                "ICD10DX-Z931"
                            ],
                            "provider_id": [
                                505,
                                131157
                            ]
                        },
                        ...
                    ]
                },
                ...
            ],
            "prospective": [
                {
                    "mem_id": 23023,
                    "gaps": [
                        {
                            "condition_category": "HCC188",
                            "confidence": 0.98871,
                            "known": 0,
                            "uccc": 1,
                            "top_indicators": [
                                "ICD10DX-K9423",
                                "ICD10DX-K9429",
                                "ICD10DX-Z931"
                            ],
                            "provider_id": [
                                505,
                                131157
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
        }
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
    if mode in ['p', 'b']:
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
    if mode in ['r', 'b']:
        vector_prior = csr_matrix(
            vstack(
                [
                    utils.build_member_input_vector(
                        dict_prior.get(mem_id, {"code": []})["code"], variables
                    )
                    for mem_id in MEMBER_LIST
                ]
            )
        )
    if mode in ['p', 'b']:
        vector_current = csr_matrix(
            vstack(
                [
                    utils.build_member_input_vector(
                        dict_prior.get(mem_id, {"code": []})["code"]
                        + dict_current.get(mem_id, {"code": []})["code"],
                        variables,
                    )
                    for mem_id in MEMBER_LIST
                ]
            )
        )

    # running machine learning
    if mode in ['r', 'b']:
        print("Running ML for retrospective analysis...")
        condition_retro = utils.run_ml(ensemble, MEMBER_LIST, vector_prior)
        condition_retro["known"] = condition_retro.apply(
            lambda x: 1
            if x.condition_category in member_known_prior.get(x.mem_id, [])
            else 0,
            axis=1,
        )
        condition_retro = condition_retro.loc[
            (condition_retro["confidence"] >= threshold)
            | (condition_retro["known"] == 1),
            :,
        ]
        condition_retro["confidence"] = condition_retro["confidence"].round(6)

    if mode in ['p', 'b']:
        print("Running ML for prospective anlysis...")
        condition_prosp = utils.run_ml(ensemble, MEMBER_LIST, vector_current)
        condition_prosp["known_current"] = condition_prosp.apply(
            lambda x: 1
            if x.condition_category in member_known_current.get(x.mem_id, [])
            else 0,
            axis=1,
        )
        condition_prosp["kown_historical"] = condition_prosp.apply(
            lambda x: 1
            if x.condition_category in member_known_prior.get(x.mem_id, [])
            else 0,
            axis=1,
        )
        condition_prosp = condition_prosp.loc[
            (condition_prosp["confidence"] >= threshold)
            | (condition_prosp["known_current"] == 1)
            | (condition_prosp["known_historical"] == 1),
            :,
        ]
        condition_prosp["confidence"] = condition_prosp["confidence"].round(6)

    # TODO: engineer this process: optimize implementation; design output
    if top_n_indicator > 0:
        if mode in ['r', 'b']:
            print("Finding retrospective indicators...")
            retro_results = utils.get_indicators(
                ensemble=ensemble,
                MEMBER_LIST=MEMBER_LIST,
                condition=condition_retro,
                vector=vector_prior,
                top_n_indicator=top_n_indicator,
                dict_prior=dict_prior,
                dict_current=dict_current,
                mappings=mappings,
                variables=variables,
            )

        if mode in ['p', 'b']:
            print("Finding prospective indicators...")
            pros_results = utils.get_indicators(
                ensemble=ensemble,
                MEMBER_LIST=MEMBER_LIST,
                condition=condition_prosp,
                vector=vector_current,
                top_n_indicator=top_n_indicator,
                dict_prior=dict_prior,
                dict_current=dict_current,
                mappings=mappings,
                variables=variables,
            )

        final_results = {
            "retrospective": retro_results if mode in ['r', 'b'] else [],
            "prospective": pros_results if mode in ['p', 'b'] else [],
        }
    else:
        final_results = {
            "retrospective": (
                utils.df_to_json(condition_retro)
                if mode in ['r', 'b']
                else []
            ),
            "prospective": (
                utils.df_to_json(condition_prosp)
                if mode in ['p', 'b']
                else []
            ),
        }

    print("End  : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return final_results


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
