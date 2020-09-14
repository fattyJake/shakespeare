# -*- coding: utf-8 -*-
###############################################################################
# Module:      training
# Description: repo of ML training functions
# Authors:     Yage Wang
# Created:     05.18.2018
###############################################################################

import os
import pickle
import itertools
import operator
import gc
from collections import Counter
from datetime import datetime

# data & machine learning libs
import pyodbc
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle

# package libs
from . import fetch_db
from . import utils


def get_training_set(year, model, payer="CD_WellPoint", server="CARABWDB06"):
    print("Fetching training dateset...")
    start_time = datetime.now()
    p_year = str(year - 1)
    codes = {}

    for y in range(year - 3, year):
        for m in range(1, 12):
            mem_date_start = datetime(y, m, 1).strftime("%Y-%m-%d")
            mem_date_end = datetime(y, m + 1, 1).strftime("%Y-%m-%d")
            sub_codes = fetch_db.batch_member_codes(
                payer=payer,
                server=server,
                date_start=p_year + "-01-01",
                date_end=p_year + "-12-31",
                mem_date_start=mem_date_start,
                mem_date_end=mem_date_end,
                model=model,
            )
            sub_codes = sub_codes.groupby("mem_id")["code"].agg(list)
            sub_codes = sub_codes.apply(lambda x: list(set(x)))
            codes.update(dict(sub_codes))
            del sub_codes
            gc.collect()

    print("Time elapase: " + str(datetime.now() - start_time))
    return codes


def update_mappings(model, sub_type_id):
    print("\nUpdating new mapping table...")
    start_time = datetime.now()
    cursor = pyodbc.connect(r"DRIVER=ODBC Driver 17 for SQL Server;" r"SERVER=MPBWDB1;" r"Trusted_Connection=yes;").cursor()

    model_name = utils.get_model_name(model)
    if ("CMS" in model_name) and (not sub_type_id):
        raise ValueError(
            "When the model is CMS(MA), sub_type_id must be provided."
        )

    if "CMS" in model_name:
        # fetch the mapping
        mapping_dict = cursor.execute(
            f"""
            SELECT ICDVersionind,UPPER(icd_code),cmsfh_hcccode
            FROM [CARA2_Processor].[dbo].[ModelCmsMapHccIcd]
            WHERE mdst_ModelSubTypeID = {sub_type_id}
                AND mv_modelversionID = {model}"""
        ).fetchall()

    elif "HHS" in model_name:
        mapping_dict = cursor.execute(
            f"""
            SELECT icdversionind,UPPER(icd_code),hhsfh_hcccode
            FROM [HIX_Processor].[dbo].[ModelHHSMapHccIcd]
            WHERE mv_modelversionID = {model}"""
        ).fetchall()

    elif "CDPS" in model_name:
        mapping_dict = cursor.execute(
            f"""
            SELECT icdversionind,UPPER(icd_code),cdps_code
            FROM [CARA2_Controller].[dbo].[ModelCdpsMapCdpsIcd]
            WHERE mv_modelversionID = {model}"""
        ).fetchall()

    else:
        raise ValueError(
            "The package only support CMS, HHS and CDPS, got {} instead.".format(
                model_name
            )
        )

    if not mapping_dict:
        raise ValueError(
            f"Empty mapping table for {model}, aborting training session."
        )

    unique_codes = sorted(
        list({f"ICD{str(i[0])}DX-" + str(i[1]) for i in mapping_dict})
    )
    new_mapping_dict = {i: [] for i in unique_codes}
    if "CDPS" in model_name:
        for i in mapping_dict:
            new_mapping_dict[f"ICD{str(i[0])}DX-" + str(i[1])].append(
                str(i[2])
            )
    else:
        for i in mapping_dict:
            new_mapping_dict[f"ICD{str(i[0])}DX-" + str(i[1])].append(
                "HCC" + str(i[2])
            )

    # dump the models
    pickle.dump(
        new_mapping_dict,
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"mappings",
                f"mapping_{model}"
                + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
            ),
            "wb",
        ),
    )

    print("Time elapase: " + str(datetime.now() - start_time))


def update_variables(codes, model, sub_type_id):
    print("\nUpdating new variable spaces...")
    start_time = datetime.now()
    mappings = list(
        pickle.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    r"mappings",
                    f"mapping_{model}"
                    + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
                ),
                "rb",
            )
        )
    )

    all_codes = []
    for k, v in codes.items():
        all_codes.extend(v)
    freq = dict(Counter(all_codes))
    freq = {k: v for k, v in freq.items() if k not in mappings}

    temp = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)[
        :15000
    ]
    variables = [i[0] for i in temp]

    pickle.dump(
        variables,
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"variables",
                f"variables_{model}"
                + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
            ),
            "wb",
        ),
    )

    del all_codes, freq
    gc.collect()

    print("Time elapase: " + str(datetime.now() - start_time))


def update_ensembles(member_codes, model, sub_type_id):
    print("\nUpdating new machine learning models...")
    start_time = datetime.now()

    param = {
        "learning_rate": 0.1,
        "n_estimators ": 500,
        "max_depth": 7,
        "min_child_weight": 1,
        "gamma": 0,
        "objective": "binary:logistic",
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "max_delta_step": 1,
        "n_jobs": -1,
        "tree_method": "hist",
        "max_bin": 512,
        "grow_policy": "lossguide",
    }

    if os.path.exists(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            r"pickle_files",
            r"ensembles",
            f"ensemble_{model}"
            + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
        )
    ):
        os.remove(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"ensembles",
                f"ensemble_{model}"
                + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
            )
        )

    mapping = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"mappings",
                f"mapping_{model}"
                + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
            ),
            "rb",
        )
    )
    HCCs = list(set(itertools.chain.from_iterable(mapping.values())))
    variables = pickle.load(
        open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                r"pickle_files",
                r"variables",
                f"variables_{model}"
                + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
            ),
            "rb",
        )
    )  # list vars as (codetype + '-' + evidencecode)
    member_conditions = {
        mem: set(
            itertools.chain.from_iterable(
                [mapping[c] for c in codes if c in mapping]
            )
        )
        for mem, codes in member_codes.items()
    }
    vec = utils.Vectorizer(variables)
    member_vectors = vec([codes for _, codes in member_codes.items()])

    ensemble = {
        k: {"classifier": None, "exemplarsPOS": None, "exemplarsNEG": None}
        for k in HCCs
    }

    # for each hcc
    for HCC in sorted(HCCs):
        try:
            ensemble = pickle.load(
                open(
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        r"pickle_files",
                        r"ensembles",
                        f"ensemble_{model}"
                        + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
                    ),
                    "rb",
                )
            )
        except FileNotFoundError:
            pass
        if ensemble[HCC]["classifier"]:
            continue

        # break up into training and testing
        labels = [1 if HCC in v else 0 for _, v in member_conditions.items()]
        if sum(labels) < 500:
            continue
        print("Training " + HCC)

        X_train, y_train = shuffle(member_vectors, labels, random_state=42)

        # save information and skip classifier on low information
        pos_train_len = sum(y_train)
        neg_train_len = len(y_train) - sum(y_train)
        ensemble[HCC]["exemplarsPOS"] = pos_train_len
        ensemble[HCC]["exemplarsNEG"] = neg_train_len

        # run ML algo
        ensemble[HCC]["classifier"] = CalibratedClassifierCV(
            XGBClassifier(**param), cv=3, method="isotonic"
        ).fit(X_train, y_train)

        # save the ensemble
        pickle.dump(
            ensemble,
            open(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    r"pickle_files",
                    r"ensembles",
                    f"ensemble_{model}"
                    + f"{'_' + str(sub_type_id) if sub_type_id else ''}",
                ),
                "wb",
            ),
        )
        del ensemble, X_train, y_train
        gc.collect()

    del member_conditions, member_vectors, vec
    gc.collect()

    print("Time elapase: " + str(datetime.now() - start_time))