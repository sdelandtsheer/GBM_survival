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
    final_df = final_df.dropna(axis=1)  # drop columns that have missing data ########### here we drop transcriptome info
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


def quantize(df, q):
    result = pd.DataFrame(0, index=df.index, columns=df.columns)
    if isinstance(q, float):
        q = [q]
    for this_q in q:
        quantiles = df.quantile(this_q)
        df_q = df > quantiles
        result = result + df_q.astype(int)

    return result

###########################################
###########################################


df, expression_data, list_of_genes = import_files()
random_genes = expression_data.reindex(np.random.permutation(expression_data.index)).iloc[:, 2000:2036]
final_df_all_genes = preprocess_and_merge(df, expression_data, list_of_genes, False)
final_df_select = preprocess_and_merge(df, expression_data, list_of_genes, True)
final_df_random = preprocess_and_merge(df, random_genes, list_of_genes, False)


y_train = final_df_all_genes.loc[:, ["bool_dead", "time"]] # make time-to-event and boolean status

X_train_all_genes = final_df_all_genes.drop(["time", "bool_dead"], axis=1).iloc[:, 4:]
X_train_select = final_df_select.drop(["time", "bool_dead"], axis=1).iloc[:, 4:]
X_train_random = final_df_random.drop(["time", "bool_dead"], axis=1).iloc[:, 4:]

X_train_all_genes_q = quantize(X_train_all_genes, 0.5)
X_train_select_q = quantize(X_train_select, 0.5)
X_train_random_q = quantize(X_train_random, 0.5)


#####################

#######################



gbm_model_all_genes = SurvivalModel(X_train=X_train_all_genes_q, y_train=y_train)
gbm_model_select = SurvivalModel(X_train=X_train_select_q, y_train=y_train)
gbm_model_random = SurvivalModel(X_train=X_train_random_q, y_train=y_train)

# gbm_model_all_genes.fit_cox_ph()
gbm_model_select.fit_cox_ph()
# gbm_model_random.fit_cox_ph()

# gbm_model_select.fit_cox_ph(mode="parsimonious")
# gbm_model_select.fit_cox_ph(mode="elastic-net")

coeffs_select = (gbm_model_select.coeffs).sort_values(ascending=False, key=abs)
# coeffs_random = (gbm_model_random.coeffs).sort_values(ascending=False, key=abs)
# coeffs_select_p = (gbm_model_select.coeffs).sort_values(ascending=False, key=abs)
# coeffs_select_en = (gbm_model_select.coeffs).sort_values(ascending=False, key=abs)

signature = {}
for this_coeff in coeffs_select.index:
    if coeffs_select[this_coeff] < 0:
        signature[this_coeff] = -1
    else:
        signature[this_coeff] = 1




X_train_select_q['combined'] = 0

for gene in signature.keys():
    X_train_select_q['combined'] = X_train_select_q['combined'] + (signature[gene] * X_train_select_q[gene])

X_train_select_q['signature'] = (X_train_select_q['combined'] < 0).astype(int)



X_train_select_q = X_train_select_q.sort_values(by='combined')
gbm_model_select = SurvivalModel(X_train=X_train_select_q, y_train=y_train)
gbm_model_select.plot_data(feature="signature")
gbm_model_select.fit_cox_ph()


#######################################
df_x = X_train_select_q
df_y = y_train
#KM curve
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
group1=df_y[df_x['signature']==1]
group2=df_y[df_x['signature']==0]
T=group1['time']
E=group1['bool_dead']
T1=group2['time']
E1=group2['bool_dead']

kmf = KaplanMeierFitter()

ax = plt.subplot(111)
ax = kmf.fit(T, E, label="Group 1-Treatment").plot(ax=ax)
ax = kmf.fit(T1, E1, label="Group 2 - Placebo").plot(ax=ax)
plt.savefig("2-groups")

#logrank_test
from lifelines.statistics import logrank_test
results=logrank_test(T,T1,event_observed_A=E, event_observed_B=E1)
results.print_summary()










scores = gbm_model_select.fit_score_features()
gbm_model_select.plot_coefficients(gbm_model_select.coeffs, 5)


for feature in gbm_model_select.X_train.columns:
    gbm_model_select.plot_data(feature="combined")







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
