# -*- coding: utf-8 -*-
###############################################################################
# Module:      visualizations
# Description: repo of tools for visualization
# Authors:     William Kinsman, Yage Wang
# Created:     11.06.2017
###############################################################################

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from . import xgb_coef

def plot_coefficients(classifier,variables,name,n=50,bottom=False,save_name=None):
    """
    Plot all or top n variables in the classifier
    @param classifier: the classifier to use in the ensemble
    @param variables: variable space to map plot axis
    @param name: HCC name of the coefficient
    @param n: returns top/bottom n variables
    @param bottom: returns bottom n variables
    
    Examples
    --------
    >>> from shakespeare.visualizations import plot_coefficients
    >>> from xgboost import XGBClassifier
    >>> xgb = XGBClassifier().fit(X_train, y_train)
    >>> plot_coefficients(xgb, variables, name='HCC22', n=100)
    """
    # initialize
    if not classifier: return
    if 'XGBClassifier' in str(classifier.__str__): coefs = xgb_coef.coef(classifier)
    if 'CalibratedClassifierCV' in str(classifier.__str__):
        coefs = [xgb_coef.coef(c.base_estimator) for c in classifier.calibrated_classifiers_]
        coefs = np.sum(coefs, axis=0) / classifier.cv
    
    if isinstance(coefs,np.ndarray): coefs = coefs.tolist()
    else: coefs = list(coefs.toarray())
    if len(variables)<n: n=len(variables)
    assert len(coefs)==len(variables),"ERROR: variable-coefficient size mismatch. Aborting."
    
    # select top n
    combined = [i for i in zip(variables,coefs)]
    combined.sort(key=lambda x:x[1], reverse=True)
    variables = [i[0] for i in combined]
    coefs = [i[1] for i in combined]
    coefs,variables = (list(t) for t in zip(*sorted(zip(coefs,variables),reverse=True)))
    if bottom: 
        variables = variables[0:n] + variables[-n:]
        coefs = coefs[0:n] + coefs[-n:]
    else:
        variables = variables[0:n]
        coefs = coefs[0:n]    
    variables = variables[::-1]
    coefs = coefs[::-1]
    
    # format variables
    codes = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes'),"rb"))
    variables = [codes[i]+'  '+i if i in codes else i for i in variables]
    
    # plot
    spacing = 3
    barwidth = 0.2
    fig = plt.figure(figsize=(5,((len(variables)+1) + 2*spacing)*barwidth))
    plt.barh(np.arange(len(variables)),coefs,align='center',alpha=0.5)
    if bottom: plt.title('Top & Bottom ' + str(int(n)) + ' Variable Coefficients')
    else: plt.title('Top ' + str(int(n)) + ' Variable Coefficients for ' + name)
    plt.xlabel('Coefficients')
    plt.yticks(np.arange(len(variables)),variables)
    plt.ylim((-1*barwidth*4,len(variables)-barwidth))
    plt.grid(linestyle='dashed',axis='x')
    plt.show()
    if save_name: fig.savefig(save_name,bbox_inches='tight')

def plot_performance(out_true,out_pred,save_name=None):
    """
    Plot ROC, Precision-Recall, Precision-Threshold
    @param out_true: list of output booleans indicicating if True
    @param out_pred: list of probabilities

    Examples
    --------
    >>> from shakespeare.visualizations import plot_performance
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> plot_performance(y_true, y_prob, None)
    """
    # get output
    precision,recall,thresholds = precision_recall_curve(out_true,out_pred)
    fpr,tpr,_  = roc_curve(out_true,out_pred)
    # roc
    fig = plt.figure(1,figsize=(18,3))
    plt.subplot(141)
    plt.plot(fpr,tpr, color='darkorange',lw=2,label='ROC (area = %0.3f)' % roc_auc_score(out_true,out_pred))
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title("ROC")
    plt.legend(loc="lower right")    
    
    # precision recall
    plt.subplot(142)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    plt.title('Precision Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot((0,1),(0.5,0.5),'k--')
    
    # precision threshold
    plt.subplot(143)
    plt.scatter(thresholds, precision[:-1], color='k',s=1)
    plt.title('Precision Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()
    plt.plot((0,1),(0.5,0.5),'k--')
    
    # calibration
    plt.subplot(144)
    fraction_of_positives,mean_predicted_value = calibration_curve(out_true,out_pred,n_bins=5)
    plt.plot(mean_predicted_value,fraction_of_positives)
    plt.title('Calibration')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.grid()
    plt.plot((0,1),'k--')
    plt.show()
    if save_name: fig.savefig(save_name,bbox_inches='tight')
	
def plot_comparison(y_true,y_score_1,y_score_2,name_1,name_2,thre=0.5,save_name=None):
    """
    Plot ROC, Precision-Recall, Precision-Threshold
    @param y_true: list of output booleans indicicating if True
    @param y_score_1: list of probabilities of model 1
    @param y_score_2: list of probabilities of model 2
    @param name_1: name of model 1
    @param name_2: name of model 2
    @param thre: threshold for point marker on the curves

    Examples
    --------
    >>> from shakespeare.visualizations import plot_comparison
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob_1 = [0.0000342, 0.99999974, 0.84367323, 0.5400342]
    >>> y_prob_2 = [0.0000093, 0.99999742, 0.99999618, 0.2400342]
    >>> plot_comparison(y_true, y_prob_1, y_prob_2, 'SVC', 'XGBoost')
    """

    fig = plt.figure(1,figsize=(5,3))

    precision1,recall1,thresholds1 = precision_recall_curve(y_true, y_score_1)
    precision2,recall2,thresholds2 = precision_recall_curve(y_true, y_score_2)
    plt.step(recall1, precision1, label=name_1, color='b', alpha=0.5,where='post')
    plt.step(recall2, precision2, label=name_2, color='r', alpha=0.5,where='post')
    ppoint1 = precision1[:-1][np.argmin(np.abs(thresholds1 - thre))]
    rpoint1 = recall1[:-1][np.argmin(np.abs(thresholds1 - thre))]
    plt.plot(rpoint1, ppoint1, 'bo', markersize=7, label='thre'+str(thre))
    ppoint2 = precision2[:-1][np.argmin(np.abs(thresholds2 - thre))]
    rpoint2 = recall2[:-1][np.argmin(np.abs(thresholds2 - thre))]
    plt.plot(rpoint2, ppoint2, 'ro', markersize=7, label='thre'+str(thre))
    plt.title('PR Compare')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.show()
    if save_name: fig.savefig(save_name,bbox_inches='tight')