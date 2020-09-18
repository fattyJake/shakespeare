# -*- coding: utf-8 -*-
###############################################################################
# Module:      utils
# Description: repo of utilities
# Authors:     Yage Wang
# Created:     11.06.2017
###############################################################################

import os
import re
import array
import operator
import itertools
import gc
import zipfile
import io
import pickle
import pyodbc
import numpy as np
import pandas as pd
import shap
import requests
from scipy.sparse import csr_matrix


class Vectorizer(object):
    """
    Build sparse dummy vector based on variable list order

    Parameters
    --------
    varibles : list
        codes that are variables as "codetype-evidencecode"

    Examples
    --------
    >>> from shakespeare.utils import Vectorizer
    >>> vec = Vectorizer(['a', 'b', 'c'])
    >>> vec([['a', 'b'], ['c']]).toarray()
    array([[1, 1, 0],
           [0, 0, 1]])
    """

    def __init__(self, variables):

        self.variables = variables
        self._cached_dict = None
        d = np.int if all(isinstance(c, int) for c in variables) else object
        self.classes_ = np.empty(len(variables), dtype=d)
        self.classes_[:] = variables


    def __call__(self, y):
        """
        Parameters
        --------
        iterable of iterables

        Return
        --------
        Sparse 2D vector of corresponding codes
        """
        class_to_index = self._build_cache()
        indices = array.array('i')
        indptr = array.array('i', [0])

        for labels in y:
            index = set()
            for label in labels:
                try:
                    index.add(class_to_index[label])
                except KeyError:
                    pass
            indices.extend(index)
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=int)

        return csr_matrix(
            (data, indices, indptr),
            shape=(len(indptr) - 1, len(class_to_index))
        )


    def _build_cache(self):
        if self._cached_dict is None:
            self._cached_dict = dict(
                zip(self.classes_, range(len(self.classes_)))
            )
        return self._cached_dict


def fast_groupby(table, k_cols, v_cols, agg=lambda x: x.tolist()):
    table = table[k_cols + v_cols]
    arr = table.sort_values(k_cols).values.T
    keys, values = np.split(arr, [len(k_cols)], axis=0)
    keys = keys.astype(int)
    ukeys, index = np.unique(keys, True, axis=1)
    arr = np.split(values, index[1:], axis=1)
    return pd.concat(
        [
            pd.DataFrame(ukeys.T, columns=k_cols)]
            + [
                pd.DataFrame({c: [agg(a[i]) for a in arr]})
                for i, c in enumerate(v_cols)
            ],
        axis=1
    )


def preprocess_table(table, target_year):
    """
    Preprocess DataFrame of raw data: specialists filtering and divide table
    into prior period and current periord as form of grouped dictionary

    Parameters
    --------
    table : pandas.DataFrame
        a table with coulumn ['mem_id', 'provider_id', 'spec_id', 'year',
        'code']

    target_year : int
        target service year

    Return
    --------
    dict_prior, dict_current
        {mem_id: {"code": [], "provider_id": []}}
    """

    SPECIALISTS = {
        3, 10, 15, 17, 22, 24, 25, 27, 29, 31, 42, 43, 44, 45, 50, 59, 60, 66,
        70, 72, 73, 74, 75, 77, 78, 79, 84, 87, 88, 89, 93, 95, 119, 121, 123,
        124, 127, 133, 140, 142, 143, 145, 147, 148, 149, 152, 157, 165, 166,
        172, 176, 177, 179, 180, 181, 186, 187, 188, 189, 191, 199, 202, 206,
        209, 210, 215, 221, 222, 224, 233, 234, 237, -1,
    }

    codes = table.code.unique().tolist()
    codes_dict = dict(enumerate(codes))
    codes_reverse_dict = {c: i for i, c in codes_dict.items()}

    table["provider_id"] = table["provider_id"].fillna(-1)
    table["spec_id"] = table["spec_id"].fillna(-1)
    table["provider_id"][~table["spec_id"].isin(SPECIALISTS)] = -1

    # indexing codes
    table.code = table.code.map(lambda x: codes_reverse_dict[x])
    table = fast_groupby(
        table,
        ["mem_id", "year", "code"],
        ["provider_id"],
        agg=lambda x: x[x > -1].tolist()
    )
    table.code = table.code.map(lambda x: codes_dict[x])
    table = fast_groupby(table, ["mem_id", "year"], ["code", "provider_id"])

    return (
        table.loc[table.year < target_year, ["mem_id", "code", "provider_id"]]
        .set_index("mem_id")
        .to_dict("index"),
        table.loc[table.year == target_year, ["mem_id", "code", "provider_id"]]
        .set_index("mem_id")
        .to_dict("index"),
    )


def run_ml(ensemble: dict, MEMBER_LIST: list, vector: csr_matrix):
    """
    ML process and post-processing

    Parameters
    --------
    ensemble : dict
        ML condition_category classifier dict

    MEMBER_LIST : list
        fixed order of member ID list

    vector : scipy.sparse.csr_matrix
        ML input sparse matrix

    Return
    --------
    df_condition : pd.DataFrame
        table with column ['mem_id', 'condition_category', 'confidence', 'known' *[,'uccc']]
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
        value_vars=list(set(df_condition.columns) & set(ensemble.keys())),
    )
    df_condition.columns = ["mem_id", "condition_category", "confidence"]
    df_condition["confidence"] = df_condition["confidence"].map(
        lambda x: np.tanh(np.power(x, 1 / 2) * 4)
        if x < 0.012091892
        else np.power(x, 1 / 5)
    )
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
        fixed order of member ID list

    condition : pandas.DataFrame
        table with column ['mem_id', 'condition_category', 'confidence', 'known' *[,'uccc']]

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
                        'condition_category': str,
                        'condition_description': str,
                        'confidence': float,
                        'known': bool,
                        *['uccc': bool,]
                        'top_indicators': list,
                        'provider_id': list
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

    condition = condition.set_index(["mem_id", "condition_category"]).to_dict("index")

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

            # prospective known_current mappings
            if (
                "known_current" in condition[(mem_id, HCC)]
                and condition[(mem_id, HCC)]["known_current"] == 1
            ):
                mapped_codes = [
                    c for c in current_codes if HCC in mappings.get(c, [])
                ]
                indices = [current_codes.index(code) for code in mapped_codes]
                pra_list = [
                    dict_current[mem_id]["provider_id"][i] for i in indices
                ]
                condition[(mem_id, HCC)]["top_indicators"] = [
                    {
                        "code_type": c.split("-", 1)[0],
                        "code": c.split("-", 1)[1],
                        "provider_id": [pra for pra in p if pra > -1],
                    }
                    for c, p in zip(mapped_codes, pra_list)
                ]
                continue

            # retrospective known mappings or prospective UCCC mappings
            if (
                condition[(mem_id, HCC)].get("known_historical", 0) == 1
                or condition[(mem_id, HCC)].get("known", 0) == 1
            ):
                mapped_codes = [
                    c for c in prior_codes if HCC in mappings.get(c, [])
                ]
                indices = [prior_codes.index(code) for code in mapped_codes]
                pra_list = [
                    dict_prior[mem_id]["provider_id"][i] for i in indices
                ]
                condition[(mem_id, HCC)]["top_indicators"] = [
                    {
                        "code_type": c.split("-", 1)[0],
                        "code": c.split("-", 1)[1],
                        "provider_id": [pra for pra in p if pra > -1],
                    }
                    for c, p in zip(mapped_codes, pra_list)
                ]
                continue

            # suspected SHAP indicators
            if "known_historical" in condition[(mem_id, HCC)]:
                codes = current_codes + prior_codes
                pra_list = (
                    dict_current.get(mem_id, {"provider_id": []})[
                        "provider_id"
                    ]
                    + dict_prior[mem_id]["provider_id"]
                )
            else:
                codes = prior_codes
                pra_list = dict_prior[mem_id]["provider_id"]
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

            pra_list = [pra_list[i] for i in indices]
            condition[(mem_id, HCC)]["top_indicators"] = [
                {
                    "code_type": c.split("-", 1)[0],
                    "code": c.split("-", 1)[1],
                    "provider_id": [pra for pra in p if pra > -1],
                }
                for c, p in zip(list(coef_dict), pra_list)
            ]

        del coef_matrix
        gc.collect()

    condition = [
        {
            "mem_id": mh_tuple[0],
            "gaps": {
                **{
                    "condition_category": mh_tuple[1],
                    "condition_description": mappings.get(mh_tuple[1], "")
                },
                **d
            }
        }
        for mh_tuple, d in condition.items()
    ]
    condition = sorted(condition, key=operator.itemgetter("mem_id"))
    condition = [
        {"mem_id": key, "gaps": [d["gaps"] for d in list(group)]}
        for key, group in itertools.groupby(
            condition, key=lambda x: x["mem_id"]
        )
    ]

    return condition


def df_to_json(condition):
    condition = condition.groupby("mem_id").agg(list).to_dict("index")
    condition = [
        {
            "mem_id": mem_id,
            "gaps": [
                {k: i[j] for j, k in enumerate(d.keys())}
                for i in zip(*list(d.values()))
            ],
        }
        for mem_id, d in condition.items()
    ]

    return condition


def get_model_name(model):
    # build model dictionary
    cursor = pyodbc.connect(r"DRIVER=ODBC Driver 17 for SQL Server;" r"SERVER=MPBWDB1;" r"Trusted_Connection=yes;").cursor()
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


def update_code_desc():
    db = pyodbc.connect(r"DRIVER=ODBC Driver 17 for SQL Server;" r"SERVER=MPBWDB1;" r"Trusted_Connection=yes;")

    code_desc = {}
    # CPT
    sql = """
        SELECT [CPTCode], [MediumDescription]
        FROM [Medref].[dbo].[CPT]
        ORDER BY CPTCode, CodeEffectiveDate DESC
    """
    cpt_codes = (
        pd.read_sql_query(sql, db)
        .groupby("CPTCode")["MediumDescription"]
        .first()
        .to_dict()
    )
    code_desc.update({"CPT-" + c: d for c, d in cpt_codes.items()})

    # DRG
    sql = """
        DECLARE @MAX_VERSION INT =
            (SELECT MAX(DRGVersion)
             FROM [Medref].[dbo].[DRG]);
        SELECT [DRGCode], [Description]
        FROM [Medref].[dbo].[DRG]
        WHERE DRGVersion = @MAX_VERSION
    """
    drg_codes = (
        pd.read_sql_query(sql, db)
        .groupby("DRGCode")["Description"]
        .first()
        .to_dict()
    )
    code_desc.update({"DRG-" + c: d for c, d in drg_codes.items()})

    # HCPCS
    sql = """
        SELECT [HCPCSCode], [ShortDescription]
        FROM [Medref].[dbo].[HCPCS]
        ORDER BY HCPCSCode, CodeEffectiveDate DESC
    """
    hcpcs_codes = (
        pd.read_sql_query(sql, db)
        .groupby("HCPCSCode")["ShortDescription"]
        .first()
        .to_dict()
    )
    code_desc.update({"HCPCS-" + c: d for c, d in hcpcs_codes.items()})

    # ICD9DX
    sql = """
        SELECT [ICD9Code], [ShortDescription]
        FROM [Medref].[dbo].[ICD9Code]
        WHERE CodeType = 'D'
        ORDER BY ICD9Code, CodeEffectiveDate DESC
    """
    icd9dx_codes = (
        pd.read_sql_query(sql, db)
        .groupby("ICD9Code")["ShortDescription"]
        .first()
        .to_dict()
    )
    code_desc.update({"ICD9DX-" + c: d for c, d in icd9dx_codes.items()})

    # ICD9PX
    sql = """
        SELECT [ICD9Code], [ShortDescription]
        FROM [Medref].[dbo].[ICD9Code]
        WHERE CodeType = 'P'
        ORDER BY ICD9Code, CodeEffectiveDate DESC
    """
    icd9px_codes = (
        pd.read_sql_query(sql, db)
        .groupby("ICD9Code")["ShortDescription"]
        .first()
        .to_dict()
    )
    code_desc.update({"ICD9PX-" + c: d for c, d in icd9px_codes.items()})

    # ICD10DX
    sql = """
        SELECT [ICD10DiagnosisCode], [ShortDescription]
        FROM [Medref].[dbo].[ICD10DiagnosisCode]
        ORDER BY ICD10DiagnosisCode, CodeEffectiveDate DESC
    """
    icd10dx_codes = (
        pd.read_sql_query(sql, db)
        .groupby("ICD10DiagnosisCode")["ShortDescription"]
        .first()
        .to_dict()
    )
    code_desc.update({"ICD10DX-" + c: d for c, d in icd10dx_codes.items()})

    # ICD10PX
    sql = """
        SELECT [ICD10ProcedureCode], [ShortDescription]
        FROM [Medref].[dbo].[ICD10ProcedureCode]
        ORDER BY ICD10ProcedureCode, CodeEffectiveDate DESC
    """
    icd10px_codes = (
        pd.read_sql_query(sql, db)
        .groupby("ICD10ProcedureCode")["ShortDescription"]
        .first()
        .to_dict()
    )
    code_desc.update({"ICD10PX-" + c: d for c, d in icd10px_codes.items()})

    # NDC9
    sql = """
        SELECT DISTINCT [NDC9Code], [DrugProductName]
        FROM [Medref].[dbo].[DimNDC]
        WHERE [DataSourceDesc] NOT LIKE 'FDA%'
            AND [NDC9Code] IS NOT NULL
        ORDER BY [NDC9Code]
    """
    ndc_codes_1 = (
        pd.read_sql_query(sql, db)
        .groupby("NDC9Code")["DrugProductName"]
        .first()
        .to_dict()
    )

    response = requests.get("https://www.accessdata.fda.gov/cder/ndctext.zip")
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall()
    ndc_codes_2 = pd.read_csv(
        z.open("product.txt"), sep="\t", header=0, encoding="ISO-8859-1"
    )
    z.close()

    ndc_codes_2.PRODUCTNDC = ndc_codes_2.PRODUCTNDC.map(ndc10_to_ndc9_product)
    ndc_codes_2.NONPROPRIETARYNAME = ndc_codes_2.NONPROPRIETARYNAME.str.lower()
    ndc_codes_2 = (
        ndc_codes_2.groupby("PRODUCTNDC")["NONPROPRIETARYNAME"].first().to_dict()
    )

    ndc_codes_1.update(ndc_codes_2)
    code_desc.update({"NDC9-" + c: d for c, d in ndc_codes_1.items()})

    pickle.dump(
        code_desc,
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"codes",
            ),
            "wb",
        ),
    )


def unique_keeping_order(iterable):
    seen = set()
    return [x for x in iterable if not (x in seen or seen.add(x))]


def ndc10_to_ndc9_product(code):
    labeler, product = code.split("-")
    if len(labeler) == 4 and len(product) == 4:
        return "0" + labeler + product
    elif len(labeler) == 5 and len(product) == 3:
        return labeler + "0" + product
    elif len(labeler) == 5 and len(product) == 4:
        return labeler + product
    else:
        return labeler + product
