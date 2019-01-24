# -*- coding: utf-8 -*-
###############################################################################
# Module:      vectorizers
# Description: repo of vectorizers for shakespeare
# Authors:     William Kinsman
# Created:     11.06.2017
###############################################################################

import scipy
import numpy as np
from datetime import datetime

def build_member_input_vector(member_codes_found,variables,dob=None,gender=None):
    """
    Build sparse dummy vector based on variable list order
    @param member_codes_found: tuple of (clientid,memberid)
    @param varibles: codes that are variables as "codetype-evidencecode"
    
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
    output = np.zeros((1,len(variables)),dtype=int)
    for i in member_codes_found:
        if i in variables: output[0,variables.index(i)] = 1
        
    # append demograpic data
    if dob and gender: output = np.hstack((output,build_demographic_vector(dob,gender)))
    return scipy.sparse.csr_matrix(output)

def build_demographic_vector(dob,gender,today=None):
    """
    @param gender: string describing gender, defaults to female
    @param dob: date of birth (YYYY-MM-DD)
    @param today (opt): uses given date for age (YYYY-MM-DD), else uses today
    """
    
    # initialization of gender (male = 1, female = 0 (defualt))
    gender = 1 if gender in ['1', 'M','m','MALE','Male','male'] else 0
    
    # initialization of dob (YYYY-MM-DD)
    if today: today = datetime.strptime(today,'%Y-%m-%d')
    else:     today = datetime.today()
    if isinstance(dob, str): dob = datetime.strptime(dob,'%Y-%m-%d')
    age = (today-dob).days/365
    
    # vectorize as numpy array of [gender,0-18,19-44,45-64,65-84,85+,Unknown,White (non-Hispanic),Black (non-Hispanic),Other,Asian/Pacific Islander,Hispanic/Latino,North American Native]
    vector = np.array([[gender,
                       age>=0  and age<18,
                       age>=18 and age<45,
                       age>=45 and age<65,
                       age>=65 and age<85,
                       age>=85,
                       ]])
    return vector
