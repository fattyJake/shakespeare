# -*- coding: utf-8 -*-
###############################################################################
# Module:      vectorizers
# Description: repo of utilities
# Authors:     Yage Wang
# Created:     11.06.2017
###############################################################################

import re
import pyodbc
import numpy as np
import pandas as pd
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


def run_ml(ensemble, MEMBER_LIST, vector, threshold):
    """
    ML process and post-processing
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
