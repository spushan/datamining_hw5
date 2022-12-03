#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.to_numpy()

k_range = [i for i in range(2, 21)]
score = []

#run kmeans testing different k values from 2 until 20 clusters
for k in k_range:

     kmeans = KMeans(n_clusters=k, random_state=0)
     kmeans.fit(X_training)
     sscore = silhouette_score(X_training, kmeans.labels_)
     score.append(sscore)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(k_range, score)
plt.xlabel('k')
plt.ylabel('silhouette_score')
plt.xticks(k_range)
plt.show()


kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X_training)
#reading the validation data (clusters) by using Pandas library
Y = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
labels = np.array(Y).reshape(1, len(Y))[0]


#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

