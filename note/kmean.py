# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:32:10 2019

@author: Jerry
"""

import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.cluster import KMeans
from sklearn import datasets
import random
iris = datasets.load_iris()
 
X = iris.data[:, :2]  # #表示我們取特徵空間中的4個維度

def get_euclidean(point1, point2):
    #point array type
    return np.sqrt(sum(pow(point1-point2,2)))
#
def get_kmean_labels(arr, n_clusters=3, max_iter=10):
    length = arr.shape[0]
    labels = np.array([-1 for _ in range(length)])
    
    #初始點
    points = [arr[p,:] for p in random.sample(list(range(length)), n_clusters)]
    
    for _ in range(max_iter):
        for i in range(length):
            
            #預設距離cluster 0最近
            short_dis = get_euclidean(arr[i,:], points[0])
            labels[i] = 0
            
            for n in range(1,n_clusters):
                dis = get_euclidean(arr[i,:], points[n])
                if dis<short_dis:
                    short_dis = dis
                    labels[i] = n
                
        points = [np.mean(X[labels==i],axis=0) for i in range(n_clusters)]
    return labels

pred = get_kmean_labels(X, n_clusters=3, max_iter=10)

#實作
x0 = X[pred == 0]
x1 = X[pred == 1]
x2 = X[pred == 2]
fig, axes = plt.subplots(2,1,figsize=(12, 8)) # 定义画布和图形
axes[0].scatter(x0[:,0], x0[:, 1], c="red", marker='o', label='label0')  
axes[0].scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
axes[0].scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')    


#套件
estimator = KMeans(n_clusters=3)  # 構造聚類器
estimator.fit(X)  # 聚類
label_pred = estimator.labels_  # 獲取聚類標籤
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
axes[1].scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  
axes[1].scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
axes[1].scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')  

