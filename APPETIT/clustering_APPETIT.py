"""Clustering procedure
2024 by
Marc Toutain, marc (at) toutain (at) unicaen (dot) fr
Jeremy lefort-Besnard, jlefortbesnard (at) tuta (dot) io

This code:
    extracts variable scores for clustering,
    standardizes them (z score),
    removes the residual for sex, age and IMC
    runs the clustering procedure (kmeans, with k=3 according to nbclust)
    saves the cleaned and standardized data with cluster affiliation for each participant in a new df
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import os


# get data
df_data = pd.read_excel("ED_prediction_scored_data_APPETIT.xlsx")
assert df_data.shape == (1053, 58) #1053 subjects, 58 variables

# Create folders to store temporary df
if not os.path.exists("created_df"):
    os.mkdir("created_df")

# Standardize data used for the clustering analysis
variable_names_included_in_clustering = [
       'Age','BMI','Psychological_motives',
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
       'F-MPS Concern with precision, order and organisation', 'sport_time']
array_scores_for_clustering = df_data[variable_names_included_in_clustering].values
array_standardized_scores_for_clustering = StandardScaler().fit_transform(array_scores_for_clustering)
assert array_standardized_scores_for_clustering.shape == (1053, 34) #1053 subjects, 35 variables

# save the standardized data into a copy of the original df
df_data_standardized = df_data.copy()
df_data_standardized[variable_names_included_in_clustering] = array_standardized_scores_for_clustering

# extract data to run the R package "nbclust" to check for best nb of cluster
df_data_standardized[variable_names_included_in_clustering].to_excel("created_df/R_data_to_run_bestClusterNb.xlsx")


# R code to apply nbclust (R package) to get best cluster nb according to 30 metrics, output is paste at the end
"""
install.packages("NbClust")
install.packages("readxl")
require("NbClust")
library(readxl)
data <- read_excel("ED_prediction/created_df/df_R_bestClusterNb.xlsx")
data = subset(data, select = c(2:36))
set.seed(42)
NbClust(data, min.nc = 2, max.nc = 8, method = 'kmeans')

R output
******************************************************************* 
* Among all indices:                                                
* 11 proposed 2 as the best number of clusters 
* 8 proposed 3 as the best number of clusters 
* 2 proposed 4 as the best number of clusters 
* 1 proposed 5 as the best number of clusters 
* 1 proposed 6 as the best number of clusters 
* 1 proposed 8 as the best number of clusters 

                   ***** Conclusion *****                            
 
* According to the majority rule, the best number of clusters is  2 
******************************************************************* 
"""

# nb cluster = 2 according to R output cluster 
output_clustering = KMeans(n_clusters=2, random_state=0).fit(df_data[variable_names_included_in_clustering].values)

df_data["cluster"] = output_clustering.labels_
df_data_standardized["cluster"] = output_clustering.labels_

df_data.to_excel("created_df/df_data_incl_cluster_labels.xlsx")
df_data_standardized.to_excel("created_df/df_data_standardized_incl_cluster_labels.xlsx")

