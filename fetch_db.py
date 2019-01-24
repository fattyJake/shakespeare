# -*- coding: utf-8 -*-
###############################################################################
# Module:      fetch_db
# Description: repo of database functions for shakespeare
# Authors:     William Kinsman, Yage Wang
# Created:     11.06.2017
###############################################################################

import re
import os
import pyodbc
import pickle

def get_members(payer, server='CARABWDB03', date_start=None, date_end=None):
    """
    Retrieve memberIDs for further training
    @param payer: the name of the payer table
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD

    Return
    --------
    List of client memberIDs

    Examples
    --------
    >>> from shakespeare.fetch_db import get_members
    >>> get_members("CD_AHM", '2016-01-01', '2016-01-05')
    [1915417,
     1915416,
     ...
     1869173,
     1869172]
    """
    # initialize
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';').cursor()
    date_start, date_end, = str(date_start), str(date_end)
    sql = """SELECT mem_id, mem_ClientMemberID
                   FROM """+payer+""".dbo.tbMember
                   WHERE mem_ClientMemberID IS NOT NULL AND dateInserted BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
                   ORDER BY mem_id DESC
                   """
    sql = re.sub("AND dateInserted BETWEEN 'None' AND 'None'", "", sql)
    cursor.execute(sql)
    return list(set([(i[0], i[1]) for i in cursor]))

def member_demographics(memberID,payer='CD_HEALTHFIRST',server='CARABWDB03'):
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';').cursor()
    cursor.execute("""DECLARE @Mem_ID INT = """ +str(memberID)+ """
                      SELECT M.gndr_Code AS Gender, M.mem_Birthday AS DOB
                      FROM """+str(payer)+""".[dbo].[tbMember] M
                      WHERE mem_id = @Mem_id
                      """)
    return [(i[0], i[1]) for i in cursor][0]

def batch_member_codes(payer='CD_HEALTHFIRST',server='CARABWDB03',memberIDs=None,date_start=None,date_end=None,file_date_lmt=None,mem_date_start=None,mem_date_end=None,get_client_id=True):
    """
    Retrieve a list of members' codes
    @param payer: the name of the payer table
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param memberIDs: a int or list of memberIDs; if None, fetch all members under the payer
    @param date_start: YYYY-MM-DD
    @param date_end: YYYY-MM-DD
    @param file_date_lmt: YYYY-MM-DD
    @param mem_date_start: YYYY-MM-DD, starting date to filter memberIDs by dateInserted
    @param mem_date_end: YYYY-MM-DD, ending date to filter memberIDs by dateInserted
    @param get_client_id: whether return member client IDs

    Return
    --------
    List of tuples (mem_id, *mem_ClientMemberID, encounter_id, Code)

    Examples
    --------
    >>> from shakespeare.fetch_db import batch_member_codes
    >>> batch_member_codes("CD_HEALTHFIRST", memberIDs=[1120565])
    [(1120565, '130008347', 'ICD9-4011'),
     (1120565, '130008347', 'CPT-73562'),
     ...
     (1120565, '130008347', 'CPT-92012'),
     (1120565, '130008347', 'ICD9-78659')]
    """
    
    # initialize
    if date_start == date_end: return []
    if memberIDs and not isinstance(memberIDs, list): memberIDs = [memberIDs]
    cursor = pyodbc.connect(r'DRIVER=SQL Server;'r'SERVER='+server+';', autocommit=True).cursor()
    
    date_start, date_end, file_date_lmt, mem_date_start, mem_date_end = str(date_start), str(date_end), str(file_date_lmt), str(mem_date_start), str(mem_date_end)

    if memberIDs:
        cursor.execute("CREATE TABLE #MemberList (mem_id INT)")
        cursor.execute('\n'.join(["INSERT INTO #MemberList VALUES ({})".format(str(member)) for member in memberIDs]))
        while cursor.nextset(): pass
        cursor.commit()
        
    # AND mem_id IN ("""+', '.join([str(mem) for mem in memberIDs])+""")
    sql = """
    SET NOCOUNT ON
    
    SELECT tbm.mem_id, mem_ClientMemberID
    INTO #Temp
    FROM """+payer+""".dbo.tbMember tbm
        INNER JOIN #MemberList m ON tbm.mem_id = m.mem_id
    WHERE mem_ClientMemberID IS NOT NULL AND dateInserted BETWEEN '"""+mem_date_start+"""' AND '"""+mem_date_end+"""'
    ORDER BY mem_id DESC
    
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, CASE icdVersionInd WHEN 9 THEN 'ICD9' WHEN 10 THEN 'ICD10' ELSE 'ICD9' END AS CodeType, ed.icd_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterDiagnosis ed ON e.enc_id = ed.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, 'CPT' AS CodeType, eCPT.cpt_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterCPT eCPT ON e.enc_id = eCPT.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, 'DRG' AS CodeType, eDRG.DRG_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterDRG eDRG ON e.enc_id = eDRG.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, 'HCPCS' AS CodeType, eHCPCS.HCPCS_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterHCPCS eHCPCS ON e.enc_id = eHCPCS.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""' 
    UNION
    SELECT e.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, e.enc_ID AS EncounterID, enc_serviceDate AS ServiceDate, CASE icdVersionInd WHEN 9 THEN 'ICD9Proc' WHEN 10 THEN 'ICD10Proc' ELSE 'ICD9Proc' END AS CodeType, eProc.icd_Code AS Code
    FROM """+payer+""".dbo.tbEncounter e WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbEncounterProcedure eProc ON e.enc_id = eProc.enc_id
    				INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
    WHERE e.enc_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    UNION
    SELECT p.mem_id AS MemberID, tp.mem_ClientMemberID as MemberClientID, p.pha_id AS EncounterID, pha_ServiceDate AS ServiceDate, 'NDC9' AS CodeType, NDC.ndcl_NDC9Code AS Code
    FROM """+payer+""".dbo.tbPharmacy p WITH(NOLOCK)
                    INNER JOIN """+payer+""".dbo.tbPharmacyNDC NDC ON p.pha_id = NDC.pha_id
    				INNER JOIN #Temp tp ON tp.mem_id = p.mem_id
    WHERE p.pha_ServiceDate BETWEEN '"""+date_start+"""' AND '"""+date_end+"""'
    """
    
    if not memberIDs: sql = re.sub("INNER JOIN \#MemberList m ON tbm\.mem\_id = m.mem\_id", "", sql)
    sql = re.sub(r"AND mem_id IN \(None\)", "", sql)
    sql = re.sub(r"AND dateInserted BETWEEN 'None' AND 'None'", "", sql)
    sql = re.sub(r"WHERE (e|p)\.[a-zA-Z\_]+ BETWEEN 'None' AND 'None'", "", sql)
    sql = re.sub(r"INNER JOIN "+payer+"\.dbo\.tbFile f WITH\(NOLOCK\) ON (e|p)\.Fil_Id = f\.Fil_id and F\.fil_StartDate <= 'None'", "", sql)

    table = cursor.execute(sql).fetchall()
    if get_client_id: return list(set([(i[0], i[1], i[2], i[4]+'-'+i[5]) for i in table]))
    else:             return list(set([(i[0], i[2], i[4]+'-'+i[5]) for i in table])) 

def member_tagging(memberID,payer,server,date_start=None,date_end=None,file_date_lmt=None,mode='MA'):
    """
    For generating training input purpose

    @param memberID: memberID (e.g. 1120565)
    @param payer: name of payer table (e.g. 'CD_HEALTHFIRST')
    @param server: CARA server on which the payer is located ('CARABWDB03')
    @param date_start: as 'YYYY-MM-DD' to get claims data from
    @param date_end: as 'YYYY-MM-DD' to get claims data to
    @param file_date_lmt: as 'YYYY-MM-DD' indicating the latest file date limit of patient codes, generally at the half of one year
    @param mode:  "MA" or "ACA"
    
    Return
    --------
    member's condition: list of HCCs
    member's codes: list of codes

    Examples
    --------
    >>> from shakespeare.fetch_db import member_tagging
    >>> member_tagging(1120565, payer="CD_HealthFirst", server="CARABWDB03")
    ['HCC108']

    ['NDC9-503830267',
        'CPT-1126F',
        'CPT-99401',
        'ICD9-4242',
        ...
        'ICD10-M216X2',
        'ICD9-V812',
        'ICD10-M1990']
    """

    # initialize
    assert mode in ['MA', 'ACA'], print("AttributeError: mode must be either 'MA' or 'ACA', {} provided instead.".format(str(mode)))
    
    codes = member_codes(memberID,payer=payer,server=server,date_start=date_start,date_end=date_end,file_date_lmt=file_date_lmt)
    if mode=='MA':  direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings'),'rb'))
    if mode=='ACA': direct_mappings = pickle.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)),r'pickle_files','codes_ICD_mappings_ACA'),'rb'))    
    
    # check for direct mappings
    conditions = []
    for i in codes:
        if i in direct_mappings and i not in conditions: conditions.append(direct_mappings[i])
        
    return conditions, codes