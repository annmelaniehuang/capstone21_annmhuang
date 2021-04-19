#utility functions
import pandas as pd
import numpy as np
import os
import datetime as dt
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import math
import datetime
from sklearn import preprocessing


def dataset_brief(df, name):
    print('Dataset {2} provided of size:\t{0:1d} rows and {1:1d} columns\n'.\
      format(df.shape[0], df.shape[1], name))
    print('List of columns in the dataset:')
    print(df.columns.tolist())
    print('missing values:')
    print((df.isnull().sum()).sort_values(ascending=False))
    print('missing values in perc%:')
    print((df.isnull().sum()/df.shape[0]).sort_values(ascending=False))
    for column in df.columns:
        print('Column {0} has {1:1d} unique numbers including NA or {2:1d} unique numbers excl. NA'.\
              format(column, 
                     df[column].nunique(dropna=False),
                     df[column].nunique(dropna=True)))


def get_dow_name(df, col_name):
    return pd.to_datetime(df.loc[:,(col_name)]).dt.day_name()


def my_odds_and_risks(df):
    n = df.shape[0]
    v = range(n)
    for rows in [[a,b] for a in v for b in v if a != b]:
        print(list(df.index[rows]))
        #print(df.iloc[rows,:])
        b, a, d, c = np.ravel(df.iloc[rows,:].to_numpy())
        print('OR/RR : {0:.2f} / {1:.2f}'.\
              format((a/c)/(b/d), df.iloc[rows,-1].values[0]/df.iloc[rows,-1].values[1]))


def pt_icu_etl(raw_icu):
    #drop duplicates, exclude rows without icu admission time
    raw_pt_icu = \
    raw_icu.query('intime.notnull() and outtime.notnull() and icustay_id!=229922 and (ttd_days >=0 or ttd_days.isnull())')
    raw_pt_icu.loc[:, ('age_bins')] =\
        pd.cut(raw_pt_icu.age_years, \
               [-math.inf,20,40,60,80,math.inf], \
               labels =['under20','20-40', '40-60','60-80', 'over80'])
    #get day of the week of ICU admission time
    raw_pt_icu.loc[:,('intime_weekday')] = get_dow_name(raw_pt_icu, 'intime')
    WKN = {'Saturday', 'Sunday'}
    #recognise weekend shifts
    raw_pt_icu.loc[:,('icu_adm_weekend')] = \
        [True if x in WKN else False for x in raw_pt_icu.intime_weekday]
    datetime_cols = {'dob', 'admittime', 'dischtime', 'intime', 'outtime', 
                     'hosp_deathtime', 'dod',}
    raw_pt_icu.loc[:, datetime_cols] = \
        raw_pt_icu.loc[:, datetime_cols].apply(lambda x: pd.to_datetime(x).dt.date, axis=0)
    raw_pt_icu.loc[:, 'ttd_bins'] = \
        pd.cut(raw_pt_icu.ttd_days.values, bins=[-math.inf,7,14,30, math.inf])
    #death in 7 days (ttd < 7) to create mortality label
    LABEL = 'standard_mortality_label'
    raw_pt_icu.loc[:, (LABEL)] = raw_pt_icu['ttd_days'] < 7
    pt_icu_cleaned = pd.get_dummies(raw_pt_icu, columns=['age_bins','intime_weekday'])
    return pt_icu_cleaned


def admission_etl_function(raw_ad):
    adms_dt_cols = {'admittime', 'dischtime', 'deathtime',}
    raw_ad.loc[:, adms_dt_cols] = \
    raw_ad.loc[:, adms_dt_cols].apply(lambda x: pd.to_datetime(x).dt.date, axis=0)
    raw_admission = raw_ad.sort_values(by=['subject_id', 'admittime']).set_index(['subject_id'])
    raw_admission.loc[:, ('prev_dischtime')] =\
        raw_admission.groupby(level=0).dischtime.shift(1)
    raw_admission.loc[:, ('tt_next_adm_days')] = \
        raw_admission.admittime - raw_admission.prev_dischtime
    raw_admission.loc[:, ('re_adm_in30d')] = \
        raw_admission.tt_next_adm_days.dt.days <= 30
    raw_admission.loc[:, ('len_of_adm')] = \
        (raw_admission.dischtime - raw_admission.admittime).dt.days
    raw_admission.loc[:, ('english_speaker')] = \
        raw_admission.groupby(level=0)['language'].apply(lambda x: 'ENGL' in list(x))
    raw_admission.loc[:, ('admission_type')] = \
        raw_admission.loc[:, ('admission_type')].apply(lambda x: x.replace(' ', '').lower())
    raw_admission.loc[:, ('insurance')] = \
        raw_admission.loc[:, ('insurance')].apply(lambda x: x.replace(' ', '').lower())
    adm_dummies = pd.get_dummies(raw_admission, 
                                 columns = ['insurance','admission_type'],
                                 prefix=['insure','adm_type'])
    cols_to_keep_admission = ('hadm_id','len_of_adm','re_adm_in30d','english_speaker',
                              'insure_government','insure_medicaid','insure_medicare',
                              'insure_private','insure_selfpay','adm_type_elective',
                              'adm_type_emergency','adm_type_newborn','adm_type_urgent')
    return adm_dummies.reset_index().loc[:, cols_to_keep_admission]


def hourly_vitals_etl(raw_vitals):
    raw_vitals_condition = \
        ((raw_vitals.spo2<0)|(raw_vitals.spo2>100))| \
        (\
         (raw_vitals.temperature<0)|(raw_vitals.temperature>108)|\
         ((raw_vitals.temperature>45)&(raw_vitals.temperature<96))\
        )| \
        ((raw_vitals.resprate<0)|(raw_vitals.resprate>196))| \
        ((raw_vitals.heartrate<0)|(raw_vitals.heartrate>480)) | \
        ((raw_vitals.sysbp<0)|(raw_vitals.sysbp>300))| \
        ((raw_vitals.diasbp<0)|(raw_vitals.diasbp>250)) | \
        ((raw_vitals.glucose<0)|(raw_vitals.glucose>1500))| \
        ((raw_vitals.meanarterialpressure<0)|(raw_vitals.meanarterialpressure>400))
    #bedside measurements, no pre-ICU data
    first_24_vital = raw_vitals[~raw_vitals_condition].query('hr>0 and hr<=24')
    #ETL - Vital
    first_24_vital.loc[first_24_vital.temperature > 43, ('temperature')] = \
    first_24_vital.query('temperature > 43').temperature.apply(lambda x: (x-32)*5/9)
    first_24_vital.loc[:, ('sys_bp_category')] =\
    pd.cut(first_24_vital.sysbp, [-math.inf, 120, 129, 139, 180, math.inf], \
           labels=('normal','elevated','HBP-stg1', 'HBP-stg2', 'HBP-Crisis'))
    first_24_vital.loc[:, ('dias_bp_category')] =\
    pd.cut(first_24_vital.diasbp, [-math.inf, 80, 89, 120, math.inf], \
           labels=('normal','HBP-stg1', 'HBP-stg2', 'HBP-Crisis'))
    first_24_vital.loc[:, ('bp_elevated')] =\
        (first_24_vital.sys_bp_category=='elevated')&(first_24_vital.dias_bp_category=='normal')
    first_24_vital.loc[:, ('bp_hbp_s1')] =\
        (first_24_vital.sys_bp_category=='HBP-stg1')|(first_24_vital.dias_bp_category=='HBP-stg1')
    first_24_vital.loc[:, ('bp_hbp_s2')] =\
        (first_24_vital.sys_bp_category=='HBP-stg2')|(first_24_vital.dias_bp_category=='HBP-stg2')
    first_24_vital.loc[:, ('bp_hyptsn_crisis')] =\
        (first_24_vital.sys_bp_category=='HBP-Crisis')|(first_24_vital.dias_bp_category=='HBP-Crisis')
    first_24_vital = \
    first_24_vital.assign(
        abnorm_spo2 = lambda x: x.spo2 < 95, # spo2 below 95 -> high risk of hypoxemia
        fever = lambda x: x.temperature > 38, # over 38C =fever   
        tachycardia = lambda x: x.heartrate > 100, #tachycardia
        bradycardia = lambda x: x.heartrate < 60,# bradycardia
        diabetes = lambda x: x.glucose >199, #assume rancdom plasma glucose test
        abnorm_map = lambda x: (x.meanarterialpressure > 100)|(x.meanarterialpressure < 60),
        #for doctors to check blood flow, resistance and pressure to supply bloody to major organs
    )
    first_24_vital_agg = \
    first_24_vital.loc[:, ('icustay_id', 'bp_elevated', 'bp_hbp_s1', 'bp_hbp_s2', 
                           'bp_hyptsn_crisis','abnorm_spo2', 'fever', 'tachycardia',
                           'bradycardia', 'diabetes', 'abnorm_map')].\
                    set_index('icustay_id').groupby(level=0).apply(sum)
    return first_24_vital_agg


def hourly_gcs_etl(raw_gcs):
    first_24_gcs = raw_gcs.query('hr>0 and hr<=24') #conscuousness
    #ETL - GCS
    first_24_gcs.loc[:, ('gcs_category')] = \
    pd.cut(raw_gcs.gcs, [0,8,12,math.inf], labels=['severe','moderate','mild'])
    first_24_gcs.loc[:, ('eye_no_resp')] = first_24_gcs.gcseyes == 1.0
    first_24_gcs.loc[:, ('motor_no_resp')] = first_24_gcs.gcsmotor == 1.0
    first_24_gcs.loc[:, ('verbal_no_resp')] = first_24_gcs.gcsverbal == 1.0
    first_24_gcs.loc[:, ('gcs_severe')] = first_24_gcs.gcs_category == 'severe'
    first_24_gcs.loc[:, ('gcs_moderate')] = first_24_gcs.gcs_category == 'moderate'

    first_24_gcs_agg = \
    first_24_gcs.loc[:, ('icustay_id', 'endotrachflag', 'eye_no_resp', 'motor_no_resp',
                         'verbal_no_resp', 'gcs_severe', 'gcs_moderate')].\
                    set_index('icustay_id').groupby(level=0).apply(sum)
    return first_24_gcs_agg


def hourly_labs_etl(raw_labs):
    first_24_labs = raw_labs.query('hr>0 and hr<=24') #blood tests, negative hours=pre-ICU
    #ETL - Labs
    # 'glucose', #diabetes
    # 'bilirubin','alaninetransaminase', 'aspartatetransaminase', #Hepatocytedamage #acute hepatitis
    # 'chloride', 'sodium', #electrolites
    # 'creatinine', 'albumin','bloodureanitrogen', #kidney functionality
    # 'hemoglobin', 'hematocrit', #anaemia
    # 'whitebloodcell',  'platelets' , #leukemia
    first_24_labs = \
    first_24_labs.assign(
        abnorm_bicarbonate = lambda x: (x.bicarbonate<23)|(x.bicarbonate>29), 
        abnorm_albumin = lambda x: (x.albumin<3.5)|(x.albumin>5), 
        abnorm_troponin = lambda x: (x.troponin>0.4),
        abnorm_bloodureanitrogen = lambda x: (x.bloodureanitrogen<7)|(x.bloodureanitrogen>20), 
        abnorm_partialpressureo2 = lambda x: (x.partialpressureo2<75)|(x.partialpressureo2>100),
        abnorm_bilirubin = lambda x: (x.bilirubin<0.1)|(x.bilirubin>1.0),
        abnorm_alt = lambda x: (x.alaninetransaminase<7)|(x.alaninetransaminase>56),
        abnorm_ast = lambda x: (x.aspartatetransaminase<5)|(x.aspartatetransaminase>40),
        abnorm_hemoglobin = lambda x: (x.hemoglobin<116)|(x.hemoglobin>166),
        abnorm_hematocrit = lambda x: (x.hematocrit<35.5)|(x.hematocrit>48.6),
        abnorm_wbc = lambda x: (x.whitebloodcell<3.4)|(x.whitebloodcell>9.6),
        abnorm_platelets = lambda x: (x.platelets<135)|(x.platelets>371),
        abnorm_sodium = lambda x: (x.sodium<135)|(x.sodium>145),
        abnorm_chloride = lambda x: (x.chloride<95)|(x.chloride>110),
        abnorm_creatinine = lambda x: (x.creatinine<0.6)|(x.creatinine>1.3),
        abnorm_glucose = lambda x: (x.glucose>199),
        abnorm_neutrophil = lambda x: (x.neutrophil<45)|(x.neutrophil>75), 
        abnorm_creactiveprotein = lambda x: (x.creactiveprotein>10),
        abnorm_lactate = lambda x: (x.lactate>1.0),
        abnorm_inr = lambda x: (x.intnormalisedratio<2)|(x.intnormalisedratio>3),
    )
    first_24_labs_agg = \
    first_24_labs.loc[:,\
                       ('icustay_id', 'abnorm_albumin', 'abnorm_bilirubin', 'abnorm_alt',
                        'abnorm_ast', 'abnorm_hemoglobin', 'abnorm_hematocrit', 'abnorm_wbc',
                        'abnorm_platelets', 'abnorm_sodium', 'abnorm_chloride',
                        'abnorm_bicarbonate', 'abnorm_troponin', 'abnorm_bloodureanitrogen',
                        'abnorm_partialpressureo2', 'abnorm_creatinine', 'abnorm_glucose',
                        'abnorm_neutrophil', 'abnorm_creactiveprotein', 'abnorm_lactate',
                        'abnorm_inr')].\
                    set_index('icustay_id').groupby(level=0).apply(sum)
    return first_24_labs_agg