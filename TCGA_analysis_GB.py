# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:04:28 2022

Glioma/Glioblastoma TCGA analysis - based on data from Ali

@author: apurva.badkas
"""

import pandas as pd
import numpy as np
from SurvivalModel_local import SurvivalModel

def import_files():
    # import files
    df = pd.read_csv('data\TCGA_TCGBiolinks_metadata.csv')
    df = df.set_index(df.columns[0])
    expression_data = pd.read_csv('data\TCGA_LGG_GBM.csv')
    expression_data = expression_data.set_index(expression_data.columns[0]).T
    # log-transform expression data
    log_expr = np.log(1 + expression_data)
    list_of_genes = pd.read_csv('data\ListOfGenes.txt')

    return df, expression_data, list_of_genes

def preprocess_and_merge(df, expression_data, list_of_genes, GenesFilter):
    if GenesFilter is True:
        genes_present = [x for x in expression_data.columns if x in list_of_genes["CommonGenes"].values]
        expression_data = expression_data[genes_present]
    df = df[["patient", "sample", "definition", "days_to_last_follow_up", "primary_diagnosis", "race", "gender",
             "vital_status", "age_at_index", "days_to_death", "paper_Survival..months.", "paper_Vital.status..1.dead.",
             "paper_IDH.status", "paper_IDH.codel.subtype",
             "paper_Transcriptome.Subtype", "paper_Pan.Glioma.RNA.Expression.Cluster",
             "paper_IDH.specific.RNA.Expression.Cluster"]]

    complete_df = pd.merge(left=df, right=expression_data, how="inner", left_index=True, right_index=True)
    # missing_patients = [x for x in expression_data.index if x in df.index]

    # select only WT glioblastomas (no mutants or normal)
    complete_df_wt = complete_df[complete_df['paper_IDH.status'] == 'WT']
    complete_df_wt_unique = complete_df_wt.drop_duplicates(subset='patient')
    complete_df_wt_unique.rename(
        columns={"paper_Survival..months.": "time", "paper_Vital.status..1.dead.": "bool_dead"}, inplace=True)
    final_df = complete_df_wt_unique.dropna(subset=["time", "bool_dead"])  # drop samples when either target is empty
    final_df = final_df.dropna(axis=1)  # drop columns that have missing data
    for col in final_df.columns:  # drop when only one value
        if len(final_df[col].unique()) == 1:
            final_df.drop(col, inplace=True, axis=1)
    final_df = final_df.drop(["patient", "sample", "vital_status"], axis=1)

    d = {"primary_diagnosis": "category",
         "race": "category",
         "gender": "category",
         "bool_dead": "bool"}
    final_df = final_df.astype(d)

    return final_df


###########################################
###########################################


df, expression_data, list_of_genes = import_files()

final_df = preprocess_and_merge(df, expression_data, list_of_genes, False)
final_dummy_df = preprocess_and_merge(df, expression_data, list_of_genes, True)



y_train = final_df.loc[:, ["time", "bool_dead"]] # make time-to-event and boolean status
y_train = y_train[y_train.columns[::-1]]

X_train = final_df.drop(["time", "bool_dead"], axis=1)
X_train_dummy = final_dummy_df.drop(["time", "bool_dead"], axis=1)
X_train_dummy = X_train_dummy.iloc[:, 4:]

quantiles = X_train_dummy.quantile()
X_train_dummy_q = X_train_dummy > quantiles
#######################

gbm_model_dummy = SurvivalModel(X_train=X_train_dummy_q, y_train=y_train)

gbm_model_dummy.fit_cox_ph()
gbm_model_dummy.plot_data(feature="ARID3A")
coeffs = gbm_model_dummy.coeffs
scores = gbm_model_dummy.fit_score_features()
gbm_model_dummy.plot_coefficients(gbm_model_dummy.coeffs, 5)



gbm_model.plot_data(feature="gender")






cols = pd.DataFrame(df.columns)

removing_IDH_mutant = df[df['paper_IDH.status'] == 'WT']

GB = removing_IDH_mutant[['barcode','patient','definition','paper_Survival..months.','days_to_death','paper_Transcriptome.Subtype']]

GB.to_csv('TCGA_metadata_subset.csv', index = False)
trans = GB.T
trans.columns = trans.iloc[0]

expression_data = pd.read_csv('TCGA_LGG_GBM.csv')
expression_data = expression_data.set_index(expression_data.columns[0])

intersection_cols = expression_data.columns & trans.columns

Sel_patients = expression_data[intersection_cols]

list_of_genes = pd.read_csv('ListOfGenes.txt')

Sel_patients2 = Sel_patients[Sel_patients.index.isin(list_of_genes['CommonGenes'])]
