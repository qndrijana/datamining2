import numpy as np
from sklearn.neighbors import KDTree
from sklearn.datasets import load_iris
from random import random
import heapq
import matplotlib.pyplot as plt
import pandas as pd

class OPTICS:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts
        self.order_list = []
        self.tree = None
        
        self.reach_dist = dict([])
        self.processed = dict([])
        
    def _get_neighbors(self, p, eps):
        return self.tree.query_radius([p], r=eps)[0]
 
    def _core_distance(self, p, eps, min_pts):
        num_neighbors = self.tree.query_radius([p], r=100.1, count_only=True)
        print(num_neighbors,min_pts)
        if num_neighbors < min_pts:
            return None
        
        neighbors = self.tree.query([p], k=min_pts)
        return neighbors[0].ravel()[-1]
    
    def update(self, N, p, seeds, eps, min_pts, core_dist):
        X = self.X
        reach_dist = self.reach_dist
        
        for j in N:
            o = X.iloc[j]
            new_reach_dist = max(core_dist, np.linalg.norm(p - o))

            if j not in reach_dist:
                reach_dist[j] = new_reach_dist
                heapq.heappush(seeds, (new_reach_dist, random(), j))
                
            else:
                if new_reach_dist < reach_dist[j]:
                    reach_dist[j] = new_reach_dist
                    for l in range(len(seeds)):
                        t = seeds[l]
                        
                        _, _, x = t
                        
                        if x == j:
                            seeds[l] = (new_reach_dist, random(), j)
                            
                    heapq.heapify(seeds)
                         
    def fit(self, X, y):
        eps = self.eps
        processed = self.processed
        order_list = self.order_list
        min_pts = self.min_pts
        
        # TODO: 0. Provera podataka
        
        self.X = X
        
        self.tree = KDTree(X) 
        
        for i in range(X.shape[0]):
            p = X.iloc[i]
            
            N = self._get_neighbors(p, eps)
            
            processed[i] = True
            order_list.append(i)

            core_dist = self._core_distance(p, eps, min_pts)
            if core_dist != None:
                                
                seeds = []
                self.update(N, p, seeds, eps, min_pts, core_dist)
                
                while len(seeds) > 0:
                    t = heapq.heappop(seeds)
                    _, _, j = t
                    q = X.iloc[j]
                    N_p = self._get_neighbors(q, eps)
                    
                    processed[j] = True
                    order_list.append(j)
                    print(order_list)
                    
                    
                    j_core_dist = self._core_distance(q, eps, min_pts)
                    
                    if j_core_dist != None:
                        self.update(N_p, q, seeds, eps, min_pts, j_core_dist)