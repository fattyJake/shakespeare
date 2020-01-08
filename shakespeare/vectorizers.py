# -*- coding: utf-8 -*-
###############################################################################
# Module:      vectorizers
# Description: repo of vectorizers for shakespeare
# Authors:     William Kinsman
# Created:     11.06.2017
###############################################################################

import scipy
import numpy as np

# from datetime import datetime


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
    return scipy.sparse.csr_matrix(output)
