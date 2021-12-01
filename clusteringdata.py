import numpy as np
from sklearn.neighbors import KDTree
from sklearn.datasets import load_iris
from random import random
import heapq
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import homogeneity_score


class ClusteringReports:
    def __init__(self):
        pass
    
   
    def hopkins(self, X, m=30):
        # TODO: Provera ulaza
        self.X = X
        
        tree = KDTree(X)
            
        X_copy = X.copy()
        np.random.shuffle(X_copy)
        X_sample = X_copy[:30]
        
        lower = np.min(X, axis=0)
        higher = np.max(X, axis=0)
        
        Y = np.random.uniform(lower.ravel(), higher.ravel(), (m, X.shape[1]))
        
        U = tree.query(Y, 2, return_distance=True)[0][:, 1].sum()
        W = tree.query(X_sample, 2, return_distance=True)[0][:, 1].sum()
        
        return U / (U + W)
    

    def silhouette(self, X, label):
        return silhouette_score(X, label)

   
    def homogeneity(self, class_labels, cluster_labels):
        return homogeneity_score(class_labels, cluster_labels)
    
    def generate_report(self, X, cluster_labels, class_labels=None):
        
        report = ''
        report += '============== REPORT ===============\n'
        report += f'Num samples: {X.shape[0]}\n'
        
        if class_labels is not None:
            report += f'Num classes: {len(np.array(class_labels.ravel()))}\n'
            
        report += f'Num attributes: {X.shape[1]}\n'
            
        report += f'Num clusters: {len(np.array(cluster_labels.ravel()))}\n'
        report += '\n'
        
        report += f'Silhouette coef.: {self.silhouette(X, cluster_labels)}\n'
        report += f'Homogeneity score: {self.homogeneity(class_labels, cluster_labels)}\n'
        
        report += f'Elapsed time: {"N/A"}\n'
        report += f'Num iterations: {"N/A"}\n'
        report += '====================================\n'
        
        f = open('output.txt', 'w')
        f.write(report)
        f.close()
        
        report_data = np.concatenate([X, cluster_labels.reshape(cluster_labels.shape[0], 1)], axis=1)
        
        columns = [f'Attribute_{i}' for i in range(1, X.shape[1] + 1)] + ['Cluster']
        
        df = pd.DataFrame(report_data, columns=columns)
        df = df.astype({ 'Cluster': 'int'})
        
        df.to_csv('output.csv', index=False)
        
        return True