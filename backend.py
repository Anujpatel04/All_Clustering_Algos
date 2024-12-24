import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pickle
import streamlit as st

data=pd.read_csv(r"C:\Users\a\VSCODE_NAREDH-IT\CLUSTERING\K_MEAN\all_clustering\Mall_Customers.csv")
st.title("Clustering algorithms:")

x=data.iloc[:,[3,4]].values

algos=['Kmean','AgglomerativeClustering']
s_algo=st.selectbox('Select an algorithm from here',algos)

if s_algo=='Kmean':
    models={"KMeans":KMeans(n_clusters=5)}
else:
    models={"AgglomerativeClustering":AgglomerativeClustering(linkage='ward',n_clusters=5)}

for models,model in models.items():
    pd=model
    y_pred=pd.fit_predict(x)
    data['Clustering'] = y_pred

#data.to_csv("New_File_With_ clustering.csv",index=False)
st.write(data)
if s_algo == "Kmean":
    plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    st.pyplot(plt)
