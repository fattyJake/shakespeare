# -*- coding: utf-8 -*-
###############################################################################
# Module:      utils
# Description: repo of utilities
# Authors:     Yage Wang
# Created:     11.06.2017
###############################################################################

import re
import operator
import itertools
import gc
import pyodbc
import numpy as np
import pandas as pd
import shap
from scipy.sparse import csr_matrix


def build_member_input_vector(member_codes_found, variables):
    """
    Build sparse dummy vector based on variable list order

    Parameters
    --------
    member_codes_found : list or iterable
        list of codes for one member
        
    varibles : list
        codes that are variables as "codetype-evidencecode"
    
    Return
    --------
    Sparse row vector of corresponding codes

    Examples
    --------
    >>> from shakespeare.vectorizers import build_member_input_vector
    >>> build_member_input_vector(['ICD10-I10'], variables)
    <1x9974 sparse matrix of type '<class 'numpy.int32'>'
        with 1 stored elements in Compressed Sparse Row format>
    """
    # vectorize
    output = np.zeros((1, len(variables)), dtype=int)
    for i in member_codes_found:
        if i in variables:
            output[0, variables.index(i)] = 1

    #    # append demograpic data
    #    if dob and gender: output = np.hstack(
    #        (output,build_demographic_vector(dob,gender))
    #     )
    return csr_matrix(output)


def preprocess_table(table, target_year):
    """
    Preprocess DataFrame of raw data: specialists filtering and divide table
    into prior period and current periord as form of grouped dictionary

    Parameters
    --------
    table : pandas.DataFrame
        a table with coulumn ['mem_id', 'pra_id', 'spec_id', 'year', 'code']
        
    target_year : int
        target service year
    
    Return
    --------
    dict_prior, dict_current
        {mem_id: {"code": [], "pra_id": []}}
    """

    SPECIALISTS = {
        3, 10, 15, 17, 22, 24, 25, 27, 29, 31, 42, 43, 44, 45, 50, 59, 60, 66,
        70, 72, 73, 74, 75, 77, 78, 79, 84, 87, 88, 89, 93, 95, 119, 121, 123,
        124, 127, 133, 140, 142, 143, 145, 147, 148, 149, 152, 157, 165, 166,
        172, 176, 177, 179, 180, 181, 186, 187, 188, 189, 191, 199, 202, 206,
        209, 210, 215, 221, 222, 224, 233, 234, 237, -1,
    }

    table["pra_id"] = table["pra_id"].fillna(-1)
    table["spec_id"] = table["spec_id"].fillna(-1)
    table = (
        table.groupby(["mem_id", "year", "code"])["pra_id", "spec_id"]
        .agg(list)
        .reset_index()
    )
    table["spec_id"] = table["spec_id"].map(
        lambda x: [spec in SPECIALISTS for spec in x]
    )
    table["pra_id"] = table.apply(
        lambda x: [
            int(pra) for i, pra in enumerate(x["pra_id"]) if x["spec_id"][i]
        ],
        axis=1,
    )
    table = (
        table.groupby(["mem_id", "year"])["code", "pra_id"]
        .agg(list)
        .reset_index()
    )

    return (
        table.loc[table.year < target_year, ["mem_id", "code", "pra_id"]]
        .set_index("mem_id")
        .to_dict("index"),
        table.loc[table.year == target_year, ["mem_id", "code", "pra_id"]]
        .set_index("mem_id")
        .to_dict("index"),
    )


def run_ml(
    ensemble: dict,
    MEMBER_LIST: list,
    vector: csr_matrix,
    threshold: float
):
    """
    ML process and post-processing

    Parameters
    --------
    ensemble : dict
        ML HCC classifier dict
        
    MEMBER_LIST : list
        fized order of member ID list
    
    vector : scipy.sparse.csr_matrix
        ML input sparse matrix
    
    threshold : float
        a float between 0 and 1 for filtering output confidence above it
    
    Return
    --------
    df_condition : pd.DataFrame
        table with column ['mem_id', 'hcc', 'confidence', 'known' *[,'uccc']]
    """

    df_condition = {"mem_id": MEMBER_LIST}
    for k, v in ensemble.items():
        if v["classifier"]:
            df_condition[k] = list(v["classifier"].predict_proba(vector)[:, 1])
    df_condition = pd.DataFrame(df_condition)
    df_condition = df_condition.dropna(axis=1, how="any")

    df_condition = pd.melt(
        df_condition,
        id_vars=["mem_id"],
        value_vars=[i for i in df_condition.columns if i.startswith("HCC")],
    )
    df_condition.columns = ["mem_id", "hcc", "confidence"]
    df_condition["confidence"] = df_condition["confidence"].map(
        lambda x: np.tanh(np.power(x, 1 / 2) * 4)
        if x < 0.012091892
        else np.power(x, 1 / 5)
    )
    df_condition = df_condition.loc[df_condition["confidence"] >= threshold, :]
    df_condition["confidence"] = df_condition["confidence"].round(6)
    # df_condition["confidence_norm"] = df_condition["confidence_norm"].round(6)

    return df_condition


def get_indicators(
    ensemble: dict,
    MEMBER_LIST: list,
    condition: pd.DataFrame,
    vector: csr_matrix,
    top_n_indicator: int,
    dict_prior: dict,
    dict_current: dict,
    mappings: dict,
    variables: list,
):
    """
    Adding supporting indicators to results for known and suspected conditions

    Parameters
    --------
    ensemble : dict
        ML HCC classifier dict
        
    MEMBER_LIST : list
        fized order of member ID list
    
    condition : pandas.DataFrame
        table with column ['mem_id', 'hcc', 'confidence', 'known' *[,'uccc']]
    
    vector : scipy.sparse.csr_matrix
        ML input sparse matrix
    
    top_n_indicator : int
        how many indicators to output for each member each HCC
    
    dict_prior : dict
        prior codes and pra_ids from `preprocess_table`

    dict_current : dict
        current codes and pra_ids from `preprocess_table`
    
    mappings : dict
        mappings from ICDs to HCCs

    variables : list
        ML variables space
    
    Return
    --------
    results : list
        [
            {
                'mem_id': int,
                'gaps': [
                    {
                        'hcc': str,
                        'confidence': float,
                        'known': bool,
                        *['uccc': bool,]
                        'top_indicators': list,
                        'pra_id': list
                    }
                ]
            }
        ]
    """

    coef_matrixes = {}
    for HCC in ensemble.keys():
        if ensemble[HCC]["classifier"] is None:
            continue

        explainer = shap.TreeExplainer(
            ensemble[HCC]["classifier"]
            .calibrated_classifiers_[0]
            .base_estimator
        )
        coef_matrix = explainer.shap_values(vector)
        coef_matrix = csr_matrix(vector.multiply(coef_matrix))
        coef_matrixes[HCC] = coef_matrix

        del explainer
        gc.collect()

    condition = condition.set_index(["mem_id", "hcc"]).to_dict("index")

    for HCC in ensemble.keys():
        if ensemble[HCC]["classifier"] is None:
            continue
        coef_matrix = coef_matrixes[HCC].toarray()

        for idx, mem_id in enumerate(MEMBER_LIST):
            if mem_id not in dict_prior:
                continue

            if (mem_id, HCC) not in condition:
                continue

            prior_codes = dict_prior.get(mem_id, {"code": []})["code"]
            current_codes = dict_current.get(mem_id, {"code": []})["code"]
            if not prior_codes:
                continue

            # prospective known mappings
            if (
                "uccc" in condition[(mem_id, HCC)]
                and condition[(mem_id, HCC)]["known"] == 1
            ):
                mapped_codes = [
                    c for c in current_codes if HCC in mappings.get(c, [])
                ]
                indices = [current_codes.index(code) for code in mapped_codes]
                pra_list = itertools.chain(
                    *[dict_current[mem_id]["pra_id"][i] for i in indices]
                )
                condition[(mem_id, HCC)]["top_indicators"] = mapped_codes
                condition[(mem_id, HCC)]["pra_id"] = list(
                    unique_keeping_order([pra for pra in pra_list if pra > -1])
                )
                continue

            # retrospective known mappings or prospective UCCC mappings
            if (
                condition[(mem_id, HCC)].get("uccc", 0) == 1
                or condition[(mem_id, HCC)]["known"] == 1
            ):
                mapped_codes = [
                    c for c in prior_codes if HCC in mappings.get(c, [])
                ]
                indices = [prior_codes.index(code) for code in mapped_codes]
                pra_list = itertools.chain(
                    *[dict_prior[mem_id]["pra_id"][i] for i in indices]
                )
                condition[(mem_id, HCC)]["top_indicators"] = mapped_codes
                condition[(mem_id, HCC)]["pra_id"] = list(
                    unique_keeping_order([pra for pra in pra_list if pra > -1])
                )
                continue

            # suspected SHAP indicators
            if "uccc" in condition[(mem_id, HCC)]:
                codes = prior_codes + current_codes
                pra_list = (
                    dict_prior[mem_id]["pra_id"]
                    + dict_current.get(mem_id, {"pra_id": []})["pra_id"]
                )
            else:
                codes = prior_codes
                pra_list = dict_prior[mem_id]["pra_id"]
            i = list(np.where(coef_matrix[idx] > 0)[0])
            coef_dict = dict(
                zip(
                    [variables[index] for index in i],
                    list(coef_matrix[idx][i]),
                )
            )
            coef_dict = dict(
                sorted(
                    coef_dict.items(), key=operator.itemgetter(1), reverse=True
                )[:top_n_indicator]
            )
            indices = [codes.index(code) for code in coef_dict]

            condition[(mem_id, HCC)]["top_indicators"] = list(coef_dict)
            pra_list = itertools.chain(*[pra_list[i] for i in indices])
            condition[(mem_id, HCC)]["pra_id"] = list(
                unique_keeping_order([pra for pra in pra_list if pra > -1])
            )

        del coef_matrix
        gc.collect()
    
    condition = [
        {"mem_id": mh_tuple[0], "gaps": {**{"hcc": mh_tuple[1]}, **d}}
        for mh_tuple, d in condition.items()
    ]
    condition = sorted(condition, key=operator.itemgetter('mem_id'))
    condition = [
        {"mem_id": key, "gaps": [d["gaps"] for d in list(group)]}
        for key, group in itertools.groupby(
            condition, key=lambda x: x['mem_id']
        )
    ]

    return condition


def df_to_json(condition):
    condition = condition.groupby('mem_id').agg(list).to_dict('index')
    condition = [
        {
            "mem_id": mem_id,
            "gaps": [
                {k: i[j] for j, k in enumerate(d.keys())}
                for i in zip(*list(d.values()))
            ]
        }
        for mem_id, d in condition.items()
    ]

    return condition


def get_model_name(model):
    # build model dictionary
    cursor = pyodbc.connect(r"DRIVER=SQL Server;" r"SERVER=MPBWDB1;").cursor()
    model_name = cursor.execute(
        """
        SELECT [mv_ModelVersionName]
        FROM [CARA2_Controller].[dbo].[ModelVersions]
        WHERE mv_IsActive=1 
        AND mv_ModelVersionID = {}""".format(
            model
        )
    ).fetchall()
    try:
        model_name = model_name[0][0]
    except:
        pass
        raise ValueError(
            f"{model} does not have a corresponding SQL table. Not building "
            "this model."
        )

    # get mapping of each model from respective tables
    cursor.close()
    return re.findall("""^[a-zA-Z]*""", model_name)[0].upper()


def unique_keeping_order(iterable):
    seen = set()
    return [x for x in iterable if not (x in seen or seen.add(x))]
