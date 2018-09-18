# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:08:11 2018

@author: pec12003
"""
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as hac

input1,input2,input3,input4,input5,input6 = np.loadtxt('Dataset 1.txt'),\
np.loadtxt('Dataset 2.txt'),np.loadtxt('Dataset 3.txt'),np.loadtxt('Dataset 4.txt'),\
np.loadtxt('Dataset 5.txt'),np.loadtxt('Dataset 6.txt')

datasets = [input1,input2,input3,input4,input5,input6]
n_clusters =4

#plot_num=1
#plt.figure(figsize=(9 * 2 + 3, 12.5))
#plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
#                    hspace=.01)
#for i_dataset, dataset in enumerate(datasets):

#    # normalize dataset for easier parameter selection
##    X = StandardScaler().fit_transform(dataset)
#    X = dataset
##    connectivity = kneighbors_graph(X,10, include_self=False)
##    connectivity = 0.5 * (connectivity + connectivity.T)
#    two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters)
#    ward = cluster.AgglomerativeClustering(
#        n_clusters=n_clusters, linkage='ward')
#    spectral = cluster.SpectralClustering(
#        n_clusters=n_clusters, eigen_solver='arpack',
#        affinity="nearest_neighbors")
#    average_linkage = cluster.AgglomerativeClustering(
#            linkage="average", n_clusters=n_clusters)
#    birch = cluster.Birch(n_clusters=n_clusters)
#    gmm = mixture.GaussianMixture(
#        n_components=n_clusters, covariance_type='full')
#    cluster_algo = (
#            ('KMeans',two_means),
#            ('Ward',ward),
#            ('AverageLinkage',average_linkage),
#            ('SpectralClustering',spectral),
#            ('Birch',birch),
#            ('GaussianMixture',gmm))
#    for name,algo in cluster_algo:
#        if name in ['Ward','AverageLinkage','DBSCAN','SpectralClustering']:
#            t0 = time.time()
#            y_pred = algo.fit_predict(X)
#            t1 = time.time()
#        else:        
#            t0 = time.time()
#            algo.fit(X)
#            t1 = time.time()
#            y_pred = algo.predict(X)
#        plt.subplot(len(datasets), len(cluster_algo),plot_num)
#        if i_dataset == 0:
#            plt.title(name, size=18)
#        plt.scatter(X[:, 0], X[:, 1],s=0.3, c=y_pred)
##        plt.xlim(-2.5, 2.5)
##        plt.ylim(-2.5, 2.5)
#        plt.xticks(())
#        plt.yticks(())
#        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
#                 transform=plt.gca().transAxes, size=15,
#                 horizontalalignment='right')
#        plot_num += 1
#
#plt.show()

#X = StandardScaler().fit_transform(input5)
X=input5
model = cluster.DBSCAN(eps=0.3)
y_label = model.fit_predict(X)
plt.scatter(input5[:,0],input5[:,1],s=0.4,c=y_label)
#model.fit(X)
#y_label = model.predict(X)
unique, counts = np.unique(y_label, return_counts=True)
y_dict = dict(zip(unique, counts))


delete_row = []
label_X3 =[]
for i in range(len(X)):
    if y_label[i]==12 or y_label[i]==27:
        if y_label[i]==12: label_X3.append(3)
        else: label_X3.append(2)
        delete_row.append(i)
X2 = np.delete(X,delete_row,0)
X3 = X[delete_row,:]

model2 = cluster.MiniBatchKMeans(
        n_clusters=n_clusters)
y_label2 = model2.fit_predict(X2)
plt.figure()
plt.scatter(X2[:,0],X2[:,1],s=0.4,c=y_label2)
plt.scatter(X3[:,0],X3[:,1],s=0.4,c=label_X3)