# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:04:28 2022

Glioma/Glioblastoma TCGA analysis - based on data from Ali

@author: apurva.badkas + Sebastien De Landtsheer
"""

# imports
import pandas as pd
import numpy as np
import random
from SurvivalModel_local import SurvivalModel
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt

# helper functions
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

    # select only WT glioblastomas (no mutants or normal)
    complete_df_wt = complete_df[complete_df['paper_IDH.status'] == 'WT']
    # drop second sample for patients
    complete_df_wt_unique = complete_df_wt.drop_duplicates(subset='patient')
    # rename important columns
    complete_df_wt_unique.rename(
        columns={"paper_Survival..months.": "time", "paper_Vital.status..1.dead.": "bool_dead"}, inplace=True)
    # drop samples when either target is empty
    final_df = complete_df_wt_unique.dropna(subset=["time", "bool_dead"])
    # drop columns that have missing data
    final_df = final_df.dropna(axis=1)
    for col in final_df.columns:  # drop when only one value
        if len(final_df[col].unique()) == 1:
            final_df.drop(col, inplace=True, axis=1)
    final_df = final_df.drop(["patient", "sample", "vital_status"], axis=1)

    # change data type of columns for compatibility with sksurv module
    d = {"primary_diagnosis": "category",
         "race": "category",
         "gender": "category",
         "bool_dead": "bool"}
    final_df = final_df.astype(d)

    return final_df


def quantize(df, q): # replace values with 1 / 0 based on threshold (same threshold for all columns)
    result = pd.DataFrame(0, index=df.index, columns=df.columns)
    if isinstance(q, float):
        q = [q]
    for this_q in q:
        quantiles = df.quantile(this_q)
        df_q = df > quantiles
        result = result + df_q.astype(int)

    return result

####################
### GBM ANALYSIS ###
####################

# load files and merge. We do it for the WHOLE transcriptomic profile
df, expression_data, list_of_genes = import_files()

final_df_all_genes = preprocess_and_merge(df, expression_data, list_of_genes, False)
y_train = final_df_all_genes.loc[:, ["bool_dead", "time"]] # make time-to-event and boolean status
X_train_all_genes = final_df_all_genes.drop(["time", "bool_dead"], axis=1).iloc[:, 4:]
X_train_all_genes_q = quantize(X_train_all_genes, 0.5)
gbm_model_all_genes = SurvivalModel(X_train=X_train_all_genes_q, y_train=y_train)
# model is too big to be fit in reasonable time.

final_df_select = preprocess_and_merge(df, expression_data, list_of_genes, True)
X_train_select = final_df_select.drop(["time", "bool_dead"], axis=1).iloc[:, 4:]
X_train_select_q = quantize(X_train_select, 0.5)
gbm_model_select = SurvivalModel(X_train=X_train_select_q, y_train=y_train)

# fitting the model with only the 35 genes of interest
gbm_model_select.fit_cox_ph()

# retrieving coefficients of the Cox model
coeffs_select = (gbm_model_select.coeffs).sort_values(ascending=False, key=abs)

# setting a signature
signature = {}
for this_coeff in coeffs_select.index:
    # retrieving the sign of the coefficient
    if coeffs_select[this_coeff] < 0:
        signature[this_coeff] = -1
    else:
        signature[this_coeff] = 1

# accumulating positive or negative 'points' depending under/over expression
X_train_select_q['combined'] = 0
for gene in signature.keys():
    X_train_select_q['combined'] = X_train_select_q['combined'] + (signature[gene] * X_train_select_q[gene])
# splitting the data in two groups depending if they have positive or negative total points
X_train_select_q['signature'] = (X_train_select_q['combined'] < 0).astype(int)


# X_train_select_q = X_train_select_q.sort_values(by='combined')
# setting-up the model again
gbm_model_select = SurvivalModel(X_train=X_train_select_q, y_train=y_train)
gbm_model_select.plot_data(feature="signature")
#gbm_model_select.fit_cox_ph() # no need to refit

#KM curve
#######################################
df_x = X_train_select_q
df_y = y_train
group1 = df_y[df_x['signature'] == 1]
group2 = df_y[df_x['signature'] == 0]
n_1 = group1.shape[0]
n_2 = group2.shape[0]
T = group1['time']
E = group1['bool_dead']
T1 = group2['time']
E1 = group2['bool_dead']

kmf = KaplanMeierFitter()

ax = plt.subplot(111)
ax = kmf.fit(T, E, label=f"Low-risk (n={n_1})").plot(ax=ax)
ax = kmf.fit(T1, E1, label=f"High-risk (n={n_2})").plot(ax=ax)
plt.savefig("signature_35_genesofinterest")

#logrank_test
from lifelines.statistics import logrank_test
results = logrank_test(T, T1, event_observed_A=E, event_observed_B=E1)
results.print_summary()
# p-val = 0.00165



### Now we do the same thing, but we select 35 genes at random.
#################################################
p_vals = []
for count in range(100):
    random_genes = expression_data.iloc[:, random.sample(range(0, expression_data.shape[1]), list_of_genes.shape[0])]
    final_df_random = preprocess_and_merge(df, random_genes, list_of_genes, False)
    X_train_random = final_df_random.drop(["time", "bool_dead"], axis=1).iloc[:, 4:]
    X_train_random_q = quantize(X_train_random, 0.5)
    gbm_model_random = SurvivalModel(X_train=X_train_random_q, y_train=y_train)
    gbm_model_random.fit_cox_ph()
    coeffs_random = (gbm_model_random.coeffs).sort_values(ascending=False, key=abs)

    signature_random = {}
    for this_coeff in coeffs_random.index:
        if coeffs_random[this_coeff] < 0:
            signature_random[this_coeff] = -1
        else:
            signature_random[this_coeff] = 1
    # uncomment this part to randomize the sign of gene contributions
    # signature_random = {}
    # for this_coeff in coeffs_random.index:
    #     if np.random.random() < 0.5:
    #         signature_random[this_coeff] = -1
    #     else:
    #         signature_random[this_coeff] = 1

    X_train_random_q['combined'] = 0
    for gene in signature_random.keys():
        X_train_random_q['combined'] = X_train_random_q['combined'] + (signature_random[gene] * X_train_random_q[gene])
    X_train_random_q['signature_random'] = (X_train_random_q['combined'] < 0).astype(int)

    X_train_random_q = X_train_random_q.sort_values(by='combined')
    gbm_model_random = SurvivalModel(X_train=X_train_random_q, y_train=y_train)
    gbm_model_random.plot_data(feature="signature_random")
    gbm_model_random.fit_cox_ph()

    #KM curve
    df_x_r = X_train_random_q
    df_y_r = y_train
    group1 = df_y_r[df_x_r['signature_random'] == 1]
    group2 = df_y_r[df_x_r['signature_random'] == 0]
    n_1 = group1.shape[0]
    n_2 = group2.shape[0]
    T = group1['time']
    E = group1['bool_dead']
    T1 = group2['time']
    E1 = group2['bool_dead']

    kmf_r = KaplanMeierFitter()

    ax2 = plt.subplot(111)
    ax2 = kmf_r.fit(T, E, label=f"Low-risk (n={n_1})").plot(ax=ax2)
    ax2 = kmf_r.fit(T1, E1, label=f"High-risk (n={n_2})").plot(ax=ax2)
    plt.savefig(f"signature_double_random_{count}")

    #logrank_test
    results_random = logrank_test(T, T1, event_observed_A=E, event_observed_B=E1)
    results_random.print_summary()
    random_pvalue = results_random.p_value
    p_vals.append(random_pvalue)

##############################################
##############################################

fig, ax = plt.subplots()
plt.hist(p_vals)
plt.savefig("pvals_2")