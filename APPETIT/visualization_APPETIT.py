"""Clustering procedure
2024 by
Marc Toutain, marc (at) toutain (at) unicaen (dot) fr
Jeremy lefort-Besnard, jlefortbesnard (at) tuta (dot) io

This code:
       extracts variable scores and clustering affiliation,
       labels and saves clusters regarding their ED risk (EAT-26 mean score)
       plots them in several different ways and saves the results as png files
"""
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os

# load data
df_data = pd.read_excel("created_df/df_data_incl_cluster_labels.xlsx")
df_data_standardized = pd.read_excel("created_df/df_data_standardized_incl_cluster_labels.xlsx")


# Create folders to store results
if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("results/visualisation"):
    os.mkdir("results/visualisation")

# set cluster names 
df_data["cluster"][df_data["cluster"] == 0] = "High risk"
df_data["cluster"][df_data["cluster"] == 1] = "Low risk"
assert(np.unique(df_data["cluster"].values).__len__() == 2) 
df_data_standardized["cluster"][df_data_standardized["cluster"] == 0] = "High risk"
df_data_standardized["cluster"][df_data_standardized["cluster"] == 1] = "Low risk"

# save a dataframe per cluster standardized data
df_high_risk_std = df_data_standardized[df_data_standardized["cluster"] == "High risk"]
df_high_risk_std.to_excel("created_df/df_high_risk_std.xlsx")
df_low_risk_std = df_data_standardized[df_data_standardized["cluster"] == "Low risk"]
df_low_risk_std.to_excel("created_df/df_low_risk_std.xlsx")
assert df_high_risk_std.shape == (len(df_data_standardized[df_data_standardized["cluster"]=="High risk"]), len(df_data_standardized.columns))

# save a dataframe per cluster original data
df_high_risk = df_data[df_data["cluster"] == "High risk"]
df_high_risk.to_excel("created_df/df_high_risk.xlsx")
df_low_risk = df_data[df_data["cluster"] == "Low risk"]
df_low_risk.to_excel("created_df/df_low_risk.xlsx")
assert df_high_risk.shape == (len(df_data[df_data["cluster"]=="High risk"]), len(df_data.columns))

variable_names_included_in_clustering = [
       'Age', 'BMI', 'Psychological_motives',
       'Interpersonal_motives', 'Health_motives', 'Body_related_motives',
       'Fitness_motives', 'BES_subscale_appearance',
       'BES_subscale_attribution', 'BES_subscale_weight', 'CDRS', 'Rosenberg',
       'HADS_anxiety', 'HADS_depression', 'EDSR_subscale_withdrawal',
       'EDSR_subscale_continuance', 'EDSR_subscale_tolerance',
       'EDSR_subscale_lack_control', 'EDSR_subscale_reduction_activities',
       'EDSR_subscale_time', 'EDSR_subscale_intention',
       'MAIA_Noticing_subscale', 'MAIA_Not-distracting_subscale',
       'MAIA_Not-Worrying_subscale', 'MAIA_Attention_regulation_subscale',
       'MAIA_Emotional_awareness_subscale', 'MAIA_Self-regulation_subscale',
       'MAIA_Body_listening_subscale', 'MAIA_Trusting_subscale',
       'F-MPS Concern over mistakes and doubts about actions', 'F-MPS Excessive concern with parents expectations and evaluation',
       'F-MPS Excessively high personal standards',
       'F-MPS Concern with precision, order and organisation', 'sport_time','cluster']
##
# plot original data
##
df_data_for_visualization = df_data[variable_names_included_in_clustering].melt(id_vars=['cluster'])
# dimension after melting = (nb_columns_before_melting -1)*nb_rows_before_melting
assert df_data_for_visualization.shape == ((df_data[variable_names_included_in_clustering].columns.__len__()-1) * df_data[variable_names_included_in_clustering].__len__(), 3)
plt.close('all')
g = sns.catplot(x="variable", y="value", hue="cluster", capsize=.2, height=10, legend = False, aspect=2, kind="bar", data=df_data_for_visualization)
plt.xticks(rotation=90, ha='right')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Variable", fontsize=14)
plt.yticks(fontsize = 12)
plt.xticks(fontsize = 12)
plt.legend(fontsize = 12, title = "Clusters", title_fontsize = 12)
g.savefig('results/visualisation/barplot_data.png', dpi=300)
plt.tight_layout()
plt.show()

##
# plot standardized data
##
df_data_for_visualization = df_data_standardized[variable_names_included_in_clustering].melt(id_vars=['cluster'])
plt.close('all')
g = sns.catplot(x="variable", y="value", hue="cluster", capsize=.2, height=10, legend = False, aspect=2, kind="bar", data=df_data_for_visualization)
plt.xticks(rotation=90, ha='right')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Variable", fontsize=14)
plt.yticks(fontsize = 12)
plt.xticks(fontsize = 12)
plt.legend(fontsize = 12, title = "Clusters", title_fontsize = 12)
g.savefig('results/visualisation/barplot_data_standardized.png', dpi=300)
plt.tight_layout()
plt.show()