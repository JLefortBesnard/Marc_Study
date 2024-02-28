"""Clustering procedure
2024 by
Marc Toutain, marc (at) toutain (at) unicaen (dot) fr
Jeremy lefort-Besnard, jlefortbesnard (at) tuta (dot) io

This code:
    extracts variable scores for clustering,
    standardizes them (z score),
    runs the clustering procedure (kmeans, with k=3 according to nbclust)
    saves the standardized data with cluster affiliation for each participant in a new df
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

df_data = pd.read_excel("ED_prediction_scored_data.xlsx")
#assert df_data.shape == (1926, 46) #1926 subjects, 46 variables

# Create folders to store temporary df
if not os.path.exists("created_df"):
    os.mkdir("created_df")

####################
# Standardize data 
####################
variable_names_included_to_standardized = [
       'Age', 'BMI', 'Psychological_motives',
       'Interpersonal_motives', 'Health_motives', 'Body_related_motives',
       'Fitness_motives', 'PSQI', 'CDRS', 'Global Self-Esteem',
       'Perceived physical value', 'Physical condition', 'Sport competence',
       'Physical Appearence', 'Physical Strength', 'HADS_anxiety',
       'HADS_depression', 'EDSR', 'EDSR_subscale_withdrawal',
       'EDSR_subscale_continuance', 'EDSR_subscale_tolerance',
       'EDSR_subscale_lack_control', 'EDSR_subscale_reduction_activities',
       'EDSR_subscale_time', 'EDSR_subscale_intention', 
       'MAIA_Noticing_subscale', 'MAIA_Not-distracting_subscale',
       'MAIA_Not-Worrying_subscale', 'MAIA_Attention_regulation_subscale',
       'MAIA_Emotional_awareness_subscale', 'MAIA_Self-regulation_subscale',
       'MAIA_Body_listening_subscale', 'MAIA_Trusting_subscale', 
       'F-MPS Concern over mistakes and doubts about actions',
       'F-MPS Excessive concern with parents expectations and evaluation',
       'F-MPS Excessively high personal standards',
       'F-MPS Concern with precision, order and organisation', 'sport_time']


data_to_std = df_data[variable_names_included_to_standardized].values
std_data = StandardScaler().fit_transform(data_to_std)
assert std_data.shape == (1926, 38) 

# save the standardized data into a copy of the original df
df_data_standardized = df_data.copy()
df_data_standardized[variable_names_included_to_standardized] = std_data


####################
# Clustering analysis 
####################
variable_names_included_in_clustering = [
       'Sex', 'BMI', 'Age', 'Psychological_motives',
       'Interpersonal_motives', 'Health_motives', 'Body_related_motives',
       'Fitness_motives', 'PSQI', 'CDRS', 'Global Self-Esteem',
       'Perceived physical value', 'Physical condition', 'Sport competence',
       'Physical Appearence', 'Physical Strength', 'HADS_anxiety',
       'HADS_depression', 'EDSR', 'EDSR_subscale_withdrawal',
       'EDSR_subscale_continuance', 'EDSR_subscale_tolerance',
       'EDSR_subscale_lack_control', 'EDSR_subscale_reduction_activities',
       'EDSR_subscale_time', 'EDSR_subscale_intention', 
       'MAIA_Noticing_subscale', 'MAIA_Not-distracting_subscale',
       'MAIA_Not-Worrying_subscale', 'MAIA_Attention_regulation_subscale',
       'MAIA_Emotional_awareness_subscale', 'MAIA_Self-regulation_subscale',
       'MAIA_Body_listening_subscale', 'MAIA_Trusting_subscale', 
       'F-MPS Concern over mistakes and doubts about actions',
       'F-MPS Excessive concern with parents expectations and evaluation',
       'F-MPS Excessively high personal standards',
       'F-MPS Concern with precision, order and organisation', 'sport_time']

# extract data to run the R package "nbclust" to check for best nb of cluster
df_data_standardized[variable_names_included_in_clustering].to_excel("created_df/R_data_to_run_bestClusterNb.xlsx")

# R code to apply nbclust (R package) to get best cluster nb according to 30 metrics, output is paste at the end
"""
install.packages("NbClust")
install.packages("readxl")
require("NbClust")
library(readxl)
data <- read_excel("/home/jlefortb/Marc_Study/STUDEAT/created_df/R_data_to_run_bestClusterNb.xlsx")
data = subset(data, select = c(2:39))

# RUN ONLY METRICS THAT WORK (DISSIMILARITY MATRIX IS NOT INVERTIBLE..)
set.seed(42)
indices <- c("kl", "ch", "hartigan", "cindex", "db", "silhouette", "duda", "pseudot2", "beale", "ratkowsky", "ball", "ptbiserial", "gap", "frey", "mcclain", "dunn", "hubert", "sdindex", "dindex", "sdbw")
# Perform clustering
get_best_nc <- function(index, data) {
  result <- NbClust(data, distance = "euclidean", min.nc = 2, max.nc = 8, method = 'kmeans', index = index)
  return(result$Best.nc)
}
best_nc_results <- sapply(indices, function(index) {
  print(index)
  get_best_nc(index, data)
})

# PRINT RESULTS
count_clusters <- function(best_nc_results) {
  # Initialize counters
  cluster_counts <- c(`2 clusters` = 0, `3 clusters` = 0, `4 clusters` = 0, `5 clusters` = 0, `6 clusters` = 0, `7 clusters` = 0, `8 clusters` = 0)

  # Iterate over best_nc_results and count occurrences
  for (result in best_nc_results) {
    cluster_counts[paste(result, "clusters")] <- cluster_counts[paste(result, "clusters")] + 1
  }

  # Print the counts
  print(cluster_counts)
}

# Example usage
# Assuming best_nc_results is the result obtained previously
count_clusters(best_nc_results)


R output
******************************************************************* 
* Among all indices:                                                
* 10 proposed 2 as the best number of clusters 
* 5 proposed 3 as the best number of clusters 
* 1 proposed 4 as the best number of clusters 
* 0 proposed 5 as the best number of clusters 
* 2 proposed 6 as the best number of clusters 
* 0 proposed 7 as the best number of clusters 
* 0 proposed 8 as the best number of clusters 

                   ***** Conclusion *****                            
 
* According to the majority rule, the best number of clusters is  2 
******************************************************************* 


"""


# nb cluster = 2 according to R output cluster 
output_clustering = KMeans(n_clusters=2, random_state=0).fit(df_data_standardized[variable_names_included_in_clustering].values)

df_data["cluster"] = output_clustering.labels_
df_data_standardized["cluster"] = output_clustering.labels_


####################
# Save and plot outputs
####################

df_data.to_excel("created_df/df_data_incl_cluster_labels.xlsx")
df_data_standardized.to_excel("created_df/df_data_standardized_incl_cluster_labels.xlsx")

# Create folders to store results
if not os.path.exists("results"):
    os.mkdir("results")
if not os.path.exists("results/visualisation"):
    os.mkdir("results/visualisation")

# set cluster names to a human readable form
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


variable_names_included_in_plotting = [
       'BMI', 'Age', 'Psychological_motives',
       'Interpersonal_motives', 'Health_motives', 'Body_related_motives',
       'Fitness_motives', 'PSQI', 'CDRS', 'Global Self-Esteem',
       'Perceived physical value', 'Physical condition', 'Sport competence',
       'Physical Appearence', 'Physical Strength', 'HADS_anxiety',
       'HADS_depression', 'EDSR', 'EDSR_subscale_withdrawal',
       'EDSR_subscale_continuance', 'EDSR_subscale_tolerance',
       'EDSR_subscale_lack_control', 'EDSR_subscale_reduction_activities',
       'EDSR_subscale_time', 'EDSR_subscale_intention', 
       'MAIA_Noticing_subscale', 'MAIA_Not-distracting_subscale',
       'MAIA_Not-Worrying_subscale', 'MAIA_Attention_regulation_subscale',
       'MAIA_Emotional_awareness_subscale', 'MAIA_Self-regulation_subscale',
       'MAIA_Body_listening_subscale', 'MAIA_Trusting_subscale', 
       'F-MPS Concern over mistakes and doubts about actions',
       'F-MPS Excessive concern with parents expectations and evaluation',
       'F-MPS Excessively high personal standards',
       'F-MPS Concern with precision, order and organisation', 'sport_time', 'cluster']


##
# plot original data
##

df_data_for_visualization = df_data[variable_names_included_in_plotting].melt(id_vars=['cluster'])
# dimension after melting = (nb_columns_before_melting -1)*nb_rows_before_melting
assert df_data_for_visualization.shape == ((df_data[variable_names_included_in_plotting].columns.__len__()-1) * df_data[variable_names_included_in_clustering].__len__(), 3)
plt.close('all')
g = sns.catplot(x="variable", y="value", hue="cluster", capsize=.2, height=10, legend = False, aspect=2, kind="bar", data=df_data_for_visualization)
plt.xticks(rotation=90, ha='right')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Variable", fontsize=14)
plt.yticks(fontsize = 12)
plt.xticks(fontsize = 12)
plt.legend(fontsize = 12, title = "Clusters", title_fontsize = 12)
plt.tight_layout()
g.savefig('results/visualisation/barplot_data.png', dpi=300)



##
# plot standardized data
##
df_data_for_visualization = df_data_standardized[variable_names_included_in_plotting].melt(id_vars=['cluster'])
plt.close('all')
g = sns.catplot(x="variable", y="value", hue="cluster", capsize=.2, height=10, legend = False, aspect=2, kind="bar", data=df_data_for_visualization)
plt.xticks(rotation=90, ha='right')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Variable", fontsize=14)
plt.yticks(fontsize = 12)
plt.xticks(fontsize = 12)
plt.legend(fontsize = 12, title = "Clusters", title_fontsize = 12)
plt.tight_layout()
g.savefig('results/visualisation/barplot_data_standardized.png', dpi=300)
plt.close('all')
