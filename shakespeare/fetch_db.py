# -*- coding: utf-8 -*-
###############################################################################
# Module:      fetch_db
# Description: repo of database functions for shakespeare
# Authors:     William Kinsman, Yage Wang
# Created:     11.06.2017
###############################################################################

import re
import pyodbc
from datetime import datetime
from .utils import get_model_name


def get_members(payer, server="CARABWDB03", date_start=None, date_end=None):
    """
    Retrieve memberIDs for further training

    Parameters
    --------
    payer : str
        name of payer table (e.g. 'CD_HEALTHFIRST')

    server : str
        CARA server on which the payer is located ('CARABWDB03')

    date_start : str, optional (default: None)
        as 'YYYY-MM-DD', starting date to filter memberIDs by dateInserted

    date_end : str, optional (default: None)
        as 'YYYY-MM-DD', ending date to filter memberIDs by dateInserted

    Return
    --------
    List of memberIDs

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
    cursor = pyodbc.connect(
        r"DRIVER=SQL Server;" r"SERVER=" + server + ";"
    ).cursor()
    date_start, date_end, = str(date_start), str(date_end)
    sql = (
        """SELECT mem_id
           FROM """
        + payer
        + """.dbo.tbMember
            WHERE dateInserted BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
        ORDER BY mem_id DESC
        """
    )
    sql = re.sub("WHERE dateInserted BETWEEN 'None' AND 'None'", "", sql)
    cursor.execute(sql)
    return list(set([(i[0], i[1]) for i in cursor]))


def batch_member_codes(
    payer: str,
    server: str,
    date_start: str,
    date_end: str,
    memberIDs: list = None,
    file_date_lmt: str = None,
    mem_date_start: str = None,
    mem_date_end: str = None,
    model: int = 63,
):
    """
    Retrieve a list of members' codes

    Parameters
    --------
    payer : str
        name of payer table (e.g. 'CD_HEALTHFIRST')

    server : str
        CARA server on which the payer is located ('CARABWDB03')

    date_start : str, optional (default: None)
        string as 'YYYY-MM-DD' to get claims data from

    date_end : str, optional (default: None)
        string as 'YYYY-MM-DD' to get claims data to

    memberIDs : list, optional (default: None)
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

    Return
    --------
    List of tuples (mem_id, pra_id, spec_id, Code)

    Examples
    --------
    >>> from shakespeare.fetch_db import batch_member_codes
    >>> batch_member_codes("CD_HEALTHFIRST", memberIDs=[1120565])
    [(1120565, '130008347', 'ICD9DX-4011'),
     (1120565, '130008347', 'CPT-73562'),
     ...
     (1120565, '130008347', 'CPT-92012'),
     (1120565, '130008347', 'ICD9DX-78659')]
    """

    # initialize
    if date_start == date_end:
        return []
    if memberIDs and not isinstance(memberIDs, list):
        memberIDs = [memberIDs]
    cursor = pyodbc.connect(
        r"DRIVER=SQL Server;" r"SERVER=" + server + ";", autocommit=True
    ).cursor()

    date_start, date_end, file_date_lmt, mem_date_start, mem_date_end = (
        str(date_start),
        str(date_end),
        str(file_date_lmt),
        str(mem_date_start),
        str(mem_date_end),
    )
    model_name = get_model_name(model)

    if memberIDs:
        cursor.execute("CREATE TABLE #MemberList (mem_id INT)")
        cursor.execute(
            "\n".join(
                [
                    f"INSERT INTO #MemberList VALUES ({str(member)})"
                    for member in memberIDs
                ]
            )
        )
        while cursor.nextset():
            pass
        cursor.commit()

    sql = (
        """
    SET NOCOUNT ON

    BEGIN TRY DROP TABLE #Temp END TRY BEGIN CATCH END CATCH
    SELECT tbm.mem_id
    INTO #Temp
    FROM """
        + payer
        + """.dbo.tbMember tbm
        INNER JOIN #MemberList m ON tbm.mem_id = m.mem_id
    WHERE dateInserted BETWEEN '"""
        + mem_date_start
        + """' AND '"""
        + mem_date_end
        + """'
    ORDER BY mem_id DESC"""
    )
    if not memberIDs:
        sql = re.sub(
            r"INNER JOIN #MemberList m ON tbm\.mem_id = m.mem_id", "", sql
        )
    sql = re.sub(r"WHERE dateInserted BETWEEN 'None' AND 'None'", "", sql)
    cursor.execute(sql)
    while cursor.nextset():
        pass
    cursor.commit()

    mrr_queue_sql = (
        "INNER JOIN CARA2_Processor.dbo.MrrResultQueue MQ ON "
        + "e.mrsq_mrrresultrunid = MQ.mrsq_mrrresultrunid AND "
        + "MQ.mrsq_DateCompleted <= '"
        + file_date_lmt
        + "'"
    )
    sql = (
        """
    BEGIN TRY DROP TABLE #mrr_temp END TRY BEGIN CATCH END CATCH
    IF OBJECT_ID('"""
        + re.sub(
            r"CD_",
            "CARA2_Results_" + ("HIX_" if "HHS" in model_name else ""),
            payer,
        )
        + """.dbo.MRRData') IS NOT NULL
        BEGIN
            SELECT e.mem_id AS MemberID,
                e.pra_id                            AS PraID,
                pra.spec_id                         AS SpecID,
                mrr_StartDate                       AS ServiceDate,
                CASE icdVersionInd
                WHEN 9 THEN 'ICD9DX' WHEN 10 THEN 'ICD10DX' ELSE 'ICD9DX' END
                                                    AS CodeType,
                UPPER(e.icd_Code)                   AS Code
            INTO #mrr_temp
            FROM """
            + re.sub(
                r"CD_",
                "CARA2_Results_" + ("HIX_" if "HHS" in model_name else ""),
                payer,
            )
            + """.dbo.MRRData e WITH(NOLOCK)
                INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
                LEFT JOIN """
            + payer
            + """.dbo.tbPractitioner pra ON pra.pra_id = e.pra_id
                            """
            + f"""{mrr_queue_sql if file_date_lmt != 'None' else ''}"""
            + """
            WHERE e.mrr_StartDate BETWEEN '"""
            + date_start
            + """' AND '"""
            + date_end
            + """'
        END"""
    )
    sql = re.sub(
        r"WHERE [ep]\.[a-zA-Z_]+ BETWEEN 'None' AND 'None'", "", sql
    )

    cursor.execute(sql)
    while cursor.nextset():
        pass
    cursor.commit()

    sql = """
    IF OBJECT_ID('tempdb..#mrr_temp') IS NULL
    BEGIN
        CREATE TABLE #mrr_temp (MemberID INT,
            PraID INT,
            SpecID INT,
            ServiceDate DATE,
            CodeType VARCHAR(50),
            Code VARCHAR(50))
    END"""
    cursor.execute(sql)
    while cursor.nextset():
        pass
    cursor.commit()

    sql = (
        """
    SELECT e.mem_id                                         AS MemberID,
        e.pra_id                                            AS PraID,
        pra.spec_id                                         AS SpecID,
        ISNULL(e.enc_DischargeDate, e.enc_ServiceDate)      AS ServiceDate,
        CASE icdVersionInd
            WHEN 9 THEN 'ICD9DX' WHEN 10 THEN 'ICD10DX' ELSE 'ICD9DX' END
                                                            AS CodeType,
        UPPER(ed.icd_Code)                                  AS Code
    FROM """
        + payer
        + """.dbo.tbEncounter e WITH(NOLOCK)
            INNER JOIN """
        + payer
        + """.dbo.tbEncounterDiagnosis ed ON e.enc_id = ed.enc_id
            INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
            LEFT JOIN """
        + payer
        + """.dbo.tbPractitioner pra ON pra.pra_id = e.pra_id
            INNER JOIN """
        + payer
        + """.dbo.tbFile f WITH(NOLOCK)
            ON f.fil_id = e.fil_id AND f.fil_StartDate <= '"""
        + file_date_lmt
        + """'
    WHERE ISNULL(e.enc_DischargeDate, e.enc_ServiceDate) BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
    UNION
    --Bring in diagnoses codes from RAPS Returns
    SELECT e.mem_id                                          AS MemberID,
        -1                                                   AS PraID,
        -1                                                   AS SpecID,
        raps_DOSfrom                                         AS ServiceDate,
        CASE WHEN raps_ICD10 is NULL THEN 'ICD9DX' else 'ICD10DX' END
                                                             AS CodeType,
        ISNULL(UPPER(raps_ICD10), UPPER(raps_ICD9))          AS Code
    FROM """
        + payer
        + """.dbo.tbMedicareRapsReturn e WITH(NOLOCK)
            INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
            INNER JOIN """
        + payer
        + """.dbo.tbFile f WITH(NOLOCK)
            ON f.fil_id = e.fil_id AND f.fil_StartDate <= '"""
        + file_date_lmt
        + """'
    WHERE e.raps_DOSfrom BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
    		AND ISNULL(e.raps_DiagError1, '') IN ('', '502')
    		AND ISNULL(e.raps_DiagError2, '') = ''
    		AND (ISNULL(e.raps_HICError, '') IN ('', '500')
    		AND ISNULL(e.raps_MBIError, '') IN ('', '503'))
    		AND ISNULL(e.risk_assessment_code_error, '') = ''
    UNION
    SELECT e.mem_id                                         AS MemberID,
        e.pra_id                                            AS PraID,
        pra.spec_id                                         AS SpecID,
        ISNULL(e.enc_DischargeDate, e.enc_ServiceDate)      AS ServiceDate,
        'CPT'                                               AS CodeType,
        eCPT.cpt_Code                                       AS Code
    FROM """
        + payer
        + """.dbo.tbEncounter e WITH(NOLOCK)
            INNER JOIN """
        + payer
        + """.dbo.tbEncounterCPT eCPT ON e.enc_id = eCPT.enc_id
            INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
            LEFT JOIN """
        + payer
        + """.dbo.tbPractitioner pra ON pra.pra_id = e.pra_id
            INNER JOIN """
        + payer
        + """.dbo.tbFile f WITH(NOLOCK)
            ON f.fil_id = e.fil_id AND f.fil_StartDate <= '"""
        + file_date_lmt
        + """'
    WHERE ISNULL(e.enc_DischargeDate, e.enc_ServiceDate) BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
    UNION
    SELECT e.mem_id                                         AS MemberID,
        e.pra_id                                            AS PraID,
        pra.spec_id                                         AS SpecID,
        ISNULL(e.enc_DischargeDate, e.enc_ServiceDate)      AS ServiceDate,
        'DRG'                                               AS CodeType,
        eDRG.DRG_Code                                       AS Code
    FROM """
        + payer
        + """.dbo.tbEncounter e WITH(NOLOCK)
            INNER JOIN """
        + payer
        + """.dbo.tbEncounterDRG eDRG ON e.enc_id = eDRG.enc_id
            INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
            LEFT JOIN """
        + payer
        + """.dbo.tbPractitioner pra ON pra.pra_id = e.pra_id
            INNER JOIN """
        + payer
        + """.dbo.tbFile f WITH(NOLOCK)
            ON f.fil_id = e.fil_id AND f.fil_StartDate <= '"""
        + file_date_lmt
        + """'
    WHERE ISNULL(e.enc_DischargeDate, e.enc_ServiceDate) BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
    UNION
    SELECT e.mem_id                                         AS MemberID,
        e.pra_id                                            AS PraID,
        pra.spec_id                                         AS SpecID,
        ISNULL(e.enc_DischargeDate, e.enc_ServiceDate)      AS ServiceDate,
        'HCPCS'                                             AS CodeType,
        eHCPCS.HCPCS_Code                                   AS Code
    FROM """
        + payer
        + """.dbo.tbEncounter e WITH(NOLOCK)
            INNER JOIN """
        + payer
        + """.dbo.tbEncounterHCPCS eHCPCS ON e.enc_id = eHCPCS.enc_id
            INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
            LEFT JOIN """
        + payer
        + """.dbo.tbPractitioner pra ON pra.pra_id = e.pra_id
            INNER JOIN """
        + payer
        + """.dbo.tbFile f WITH(NOLOCK)
            ON f.fil_id = e.fil_id AND f.fil_StartDate <= '"""
        + file_date_lmt
        + """'
    WHERE ISNULL(e.enc_DischargeDate, e.enc_ServiceDate) BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
    UNION
    SELECT e.mem_id                                         AS MemberID,
        e.pra_id                                            AS PraID,
        pra.spec_id                                         AS SpecID,
        ISNULL(e.enc_DischargeDate, e.enc_ServiceDate)      AS ServiceDate,
        CASE icdVersionInd
            WHEN 9 THEN 'ICD9PX' WHEN 10 THEN 'ICD10PX' ELSE 'ICD9PX' END
                                                            AS CodeType,
        eProc.icd_Code                                      AS Code
    FROM """
        + payer
        + """.dbo.tbEncounter e WITH(NOLOCK)
            INNER JOIN """
        + payer
        + """.dbo.tbEncounterProcedure eProc ON e.enc_id = eProc.enc_id
            INNER JOIN #Temp tp ON tp.mem_id = e.mem_id
            LEFT JOIN """
        + payer
        + """.dbo.tbPractitioner pra ON pra.pra_id = e.pra_id
            INNER JOIN """
        + payer
        + """.dbo.tbFile f WITH(NOLOCK)
            ON f.fil_id = e.fil_id AND f.fil_StartDate <= '"""
        + file_date_lmt
        + """'
    WHERE ISNULL(e.enc_DischargeDate, e.enc_ServiceDate) BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
    UNION
    SELECT p.mem_id                                         AS MemberID,
        p.pra_id                                            AS PraID,
        pra.spec_id                                         AS SpecID,
        pha_ServiceDate                                     AS ServiceDate,
        'NDC9'                                              AS CodeType,
        NDC.ndcl_NDC9Code                                   AS Code
    FROM """
        + payer
        + """.dbo.tbPharmacy p WITH(NOLOCK)
                    INNER JOIN """
        + payer
        + """.dbo.tbPharmacyNDC NDC ON p.pha_id = NDC.pha_id
            INNER JOIN #Temp tp ON tp.mem_id = p.mem_id
            LEFT JOIN """
        + payer
        + """.dbo.tbPractitioner pra ON pra.pra_id = p.pra_id
            INNER JOIN """
        + payer
        + """.dbo.tbFile f WITH(NOLOCK)
            ON f.fil_id = p.fil_id AND f.fil_StartDate <= '"""
        + file_date_lmt
        + """'
    WHERE p.pha_ServiceDate BETWEEN '"""
        + date_start
        + """' AND '"""
        + date_end
        + """'
    UNION
    SELECT * FROM #mrr_temp
    """
    )
    sql = re.sub(
        r"WHERE [ep]\.[a-zA-Z_]+ BETWEEN 'None' AND 'None'", "", sql
    )
    sql = re.sub(
        r"WHERE ISNULL\(e\.enc_DischargeDate, e\.enc_ServiceDate\) BETWEEN "
        + r"'None' AND 'None'",
        "",
        sql,
    )
    sql = re.sub(
        r"INNER JOIN "
        + payer
        + r"\.dbo\.tbFile f WITH\(NOLOCK\)\s+ON f\.fil_id = [ep]\.fil_id AND "
        + r"f\.fil_StartDate <= 'None'",
        "",
        sql,
    )

    table = cursor.execute(sql).fetchall()
    cursor.close()

    table = list(
        set([(i[0], i[1], i[2], i[4] + "-" + i[5]) for i in table])
    )
    return table


def batch_member_monthly_report(
    payer="CD_HEALTHFIRST",
    server="CARABWDB03",
    memberIDs=None,
    year_month=None
):
    """
    Retrieve MMR info identified by CMS

    Parameters
    --------
    payer : str
        name of payer table (e.g. 'CD_HEALTHFIRST')

    server : str
        CARA server on which the payer is located ('CARABWDB03')

    memberIDs : list
        List of member IDs to fetch

    year_month : str, optional (default: None)
        as 'YYYY-MM', the payment month (or current month)

    Return
    --------
    mmr_table : list
        list of lists with MMR information

    Examples
    --------
    >>> from shakespeare.fetch_db import batch_member_monthly_report
    >>> batch_member_monthly_report(year_month='2019-06')
    [[1962093, 1.024999976158142, 1, 81, 0, 0, 0, 0],
     [1962157, 1.0720000267028809, 0, 67, 0, 0, 0, 0],
     ...,
     [2102341, 0.45500001311302185, 0, 66, 0, 0, 0, 0],
     [2102375, 0.875, 1, 67, 0, 0, 0, 0]]
    """
    # initialize
    cursor = pyodbc.connect(
        r"DRIVER=SQL Server;" r"SERVER=" + server + ";"
    ).cursor()
    if year_month:
        year_month = str(year_month)
    else:
        year_month = str(datetime.today().year) + '-06'

    if memberIDs:
        cursor.execute("CREATE TABLE #Temp (mem_id INT)")
        cursor.execute(
            "\n".join(
                [
                    "INSERT INTO #Temp VALUES ({})".format(str(member))
                    for member in memberIDs
                ]
            )
        )
        while cursor.nextset():
            pass
        cursor.commit()

    sql = (
        """SELECT mmr.[mem_id]                              AS mem_id,
            [mmr_FactorA]                                   AS risk_score,
            [gndr_Code]                                     AS gender,
            DATEDIFF(hour,mmr_Birthday,'""" + year_month
            + """-01')/8766                                 AS age,
            [mmr_Hospice]                                   AS hospice,
            [mmr_ESRD]                                      AS ESRD,
            [mmr_AgedDisabledMSP]                           AS disabled,
            [mmr_Institutional]                             AS institutional
        FROM """ + payer + """.[dbo].[tbMedicareMMR] mmr
            LEFT JOIN #Temp t ON mmr.mem_id = t.mem_id
        WHERE mmr_AdjustReasonCode = 00 AND mmr_PmtMonth = '"""
        + year_month.replace('-', '') + """'"""
    )
    if not memberIDs:
        sql = re.sub(
            r"LEFT JOIN #Temp t ON mmr\.mem_id = t\.mem_id", "", sql
        )
    cursor.execute(sql)
    return list(
        [
            [
                i[0],
                i[1],
                0 if i[2] == "M" else 1,
                int(i[3]),
                1 if i[4].strip() else 0,
                1 if i[5].strip() else 0,
                1 if i[6].strip() == 'Y' else 0,
                1 if i[7].strip() else 0
            ]
            for i in cursor
            if i[0]
        ]
    )