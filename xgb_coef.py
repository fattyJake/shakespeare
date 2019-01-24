# -*- coding: utf-8 -*-
###############################################################################
# Module:      xgb_coef
# Description: repo of tools for getting coefficient of XGBClassifier
# Authors:     Yage Wang
# Created:     3.21.2017
###############################################################################

import numpy as np

def coef(xgb, metric='gain'):
	"""
    Customized function to get XGBClassifier coefficient
    @param xgb: the classifier to use for getting coefficience
    @param metric: type of measurement of coefficiencts
        - The "gain" implies the relative contribution of the corresponding
          feature to the model calculated by taking each feature's contribution
          for each tree in the model. A higher value of this metric when
          compared to another feature implies it is more important for
          generating a prediction.
        - The "cover" metric means the relative number of observations related
          to this feature. For example, if you have 100 observations, 4
          features and 3 trees, and suppose feature1 is used to decide the leaf
          node for 10, 5, and 2 observations in tree1, tree2 and tree3
          respectively; then the metric will count cover for this feature as
          10+5+2 = 17 observations. This will be calculated for all the 4
          features and the cover will be 17 expressed as a percentage for all
          features' cover metrics.
        - The "weight" is the percentage representing the relative number of
          times a particular feature occurs in the trees of the model. In the
          above example, if feature1 occurred in 2 splits, 1 split and 3 splits
          in each of tree1, tree2 and tree3; then the weightage for feature1
          will be 2+1+3 = 6. The frequency for feature1 is calculated as its
          percentage weight over weights of all features.
        
        The "gain" is the most relevant attribute to interpret the importance
        of features.

    Return
    --------
    Array of shape = [n_features]

    Examples
    --------
    >>> from shakespeare.xgb_coef import coef
    >>> from xgboost import XGBClassifier
    >>> xgb = XGBClassifier().fit(X_train, y_train)
    >>> coef(xgb)
    array([0.00062452, 0.00106926, 0.00028839, ..., 0.        , 0.        ,
       0.        ], dtype=float32)
	"""
	b = xgb.get_booster()
	fs = b.get_score(importance_type=metric)
	all_features = [fs.get(f, 0.) for f in b.feature_names]
	all_features = np.array(all_features, dtype=np.float32)
	return all_features / all_features.sum()