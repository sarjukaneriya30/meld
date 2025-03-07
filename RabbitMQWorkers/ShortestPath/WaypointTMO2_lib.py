import os
import sys

#########
from graphCreator8_Libs import *  # custom classes

###########
data_path = './'

#picklefile = open('C:\\Users\\nickkong\\KMPG\\WorkShop\\Data\\BAY2Obj8', 'rb')
picklefile = open(data_path + 'BAY2Obj8', 'rb')
BAY2Obj_loaded = pickle.load(picklefile)
picklefile.close()
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path  # include dijkstra
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import AgglomerativeClustering

import math
import itertools
from itertools import permutations
from ast import literal_eval
import random
import time
import pickle
import json
import os
import decimal
import copy
import os
import sys


############################################
# class BAY2_WaypointTMO

class BAY2_WaypointTMO:
    # Initialize the class. Here nodes is the list of major nodes (the nodes in road intersptions)
    def __init__(self, name='',BAYs=None):
        self.name = name
        self.BAYs = BAYs
        self.max_aisle = 43
        self.min_aisle = 1
        self.StorageAisles = {3:-1,4:6,6:4,7:9,9:7,10:-1,17:-1,18:20,20:18,21:23,23:21,24:26,26:24,\
                              27:29,29:27,34:-1,35:37,37:35,38:40,40:38,41:-1}  # pair of ailses sharing the same track/rail

        # equvilant (physical) locations for {n1,n2,...,n36}. In waypoint, the aisle numbers of n1, ...,n39 are needed.
        self.eqLocs = {'n1':(84, 3),'n2':(73, 3),'n3':(61, 3),'n4':(84, 4),'n5':(73, 4),'n6':(61, 4),'n7':(84, 7),\
                       'n8':(73, 7),'n9':(61, 7),'n10':(91, 10),'n11':(84, 10),'n12':(73, 10),'n13':(61, 10),'n14':(51, 10),\
                       'n15':(86, 18),'n16':(61, 18),'n17':(86, 21),'n18':(61, 21),'n19':(86, 24),'n20':(61, 24),\
                       'n21':(79, 29, 'S'),'n22':(61, 29, 'S'),'n23':(91, 34),'n24':(79, 29, 'N'),'n25':(70, 29, 'N'),\
                       'n26':(61, 34),'n27':(51, 34),'n28':(84, 37),'n29':(73, 37),'n30':(61, 37),'n31':(84, 40),\
                       'n32':(73, 40),'n33':(61, 40),'n34':(84, 41),'n35':(73, 41),'n36':(61, 41)}
        self.aisleDistanceMatrix = None
        self.aisleDistanceMatrix_cur = None

    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name

    # branch_Node represetation
    def __repr__(self):
        return ('({0},{1})'.format('name = ',self.name))

    # create BAY2 (storage) aisle distance matrix
    def aisleDistanceMatrixCreator(self):
        self.aisleDistanceMatrix = pd.DataFrame(data=np.empty(shape=(len(self.StorageAisles.keys()),len(self.StorageAisles.keys()))),\
                                                columns=[a for a in self.StorageAisles.keys()])
        self.aisleDistanceMatrix.index = self.aisleDistanceMatrix.columns
        self.aisleDistanceMatrix[0:][0:] = np.nan
        np.fill_diagonal(self.aisleDistanceMatrix.values, 0)

        # calcualate values of self.aisleDistanceMatrix
        storage_aisles = list(self.StorageAisles.keys()).copy()
        for a1 in self.StorageAisles.keys():
            storage_aisles.remove(a1)
            for a2 in storage_aisles:
                d = abs(int(a1) - int(a2))
                self.aisleDistanceMatrix[a1][a2] = d
                self.aisleDistanceMatrix[a2][a1] = d

        # calcualte distance between ajacent aisles such as aisle 4 and aisle 6
        for a in self.StorageAisles.keys():
            if self.StorageAisles[a] != -1:
                self.aisleDistanceMatrix[a][self.StorageAisles[a]] = 0.5
                self.aisleDistanceMatrix[self.StorageAisles[a]][a] = 0.5



    # given (agent) starting point agentLoc, find next aisle to visit
    # for example, agentLoc = str((67, 29, 'N')) or 'n1'
    #              items = ['(78, 41)', '(65, 41)','(86, 40)','(80, 35)', '(91, 35)', '(75, 38)', '(67, 37)']
    def nextAisle(self,agentLoc,items):
        try:
            #print('agentLoc in 111 = ', agentLoc, type(agentLoc))
            assert self.min_aisle <= int(literal_eval(agentLoc)[1]) <= self.max_aisle
            #print('agentLoc in 222 = ', agentLoc)

            for i in items:
                assert int(literal_eval(i)[1]) in self.StorageAisles.keys()

            # find optimal aisle ret_aisle
            dif = sys.maxsize
            ret_aisle = -1
            for i in items:
                aisle = int(literal_eval(i)[1])
                #print('aisle = ', aisle, '   dif = ', dif)
                if dif > abs(aisle-int(literal_eval(agentLoc)[1])):
                    dif = abs(aisle-int(literal_eval(agentLoc)[1]))
                    ret_aisle = aisle

            # find side
            side = ''
            if ret_aisle == 29: # only those locations in aisle 29 has two 'sides' ('S','N')
                if int(literal_eval(agentLoc)[1]) == 29:
                    side = literal_eval(agentLoc)[2]
                elif int(literal_eval(agentLoc)[1]) > 29:
                    side = 'S'
                else:
                    side = 'N'

            return (ret_aisle,side)
        except:
            print('error in startingItemLoc(...): check if locations are out of range ...')


    # given an aisle, find all item storage locations in this aisle
    # here, for example,
    #           aisle = (41, '')  or (29, 'N'), generally aisle = (aisle,side)
    #           items = ['(78, 41)', '(65, 41)','(86, 40)','(80, 35)', '(91, 35)', '(75, 38)', '(67, 37)']
    def aisleItemLocs(self,aisle,items=[]):
        assert self.min_aisle <= aisle[0] <= self.max_aisle

        if aisle[0] != 29:
            return [x for x in items if aisle[0] == int(literal_eval(x)[1])]
        else: # here, aisle[0] = 29
            #print('items = ', items)
            return [str((literal_eval(x)[0], literal_eval(x)[1], aisle[1])) for x in items if aisle[0] == int(literal_eval(x)[1])]

    # given an agent location agentLoc and an aisleItems returned by aisleItemLocs(...),
    # find optimal item sequence (in aisleItems).
    def aisleItemsOptimalSeq(self,agentLoc,aisleItems):
        assert self.min_aisle <= int(literal_eval(agentLoc)[1]) <= self.max_aisle

        maxCol = max([int(literal_eval(x)[0]) for x in aisleItems])
        minCol = min([int(literal_eval(x)[0]) for x in aisleItems])
        agentCol = int(literal_eval(agentLoc)[0])

        if abs(agentCol - maxCol) <= abs(agentCol - minCol):
            retCol = maxCol
        else:
            retCol = minCol

        nextItem = [x for x in aisleItems if retCol == int(literal_eval(x)[0])][0]
        aisleItems_c = aisleItems.copy()
        aisleItems_c.remove(nextItem)

        self.BAYs.update_cur_DistMatrix(leafnodeNames=aisleItems)
        OptSeq = self.BAYs.greedy(start_pt=nextItem,items=aisleItems_c)

        return [agentLoc] + OptSeq

    # given an agent location agentLoc and items (locations),
    # find Waypoint optimal item sequence
    # for example, agent = ('agent2', '(51, 43)') or ('agent1', '(51, 29, 'N')')
    #              items = ['(78, 41)', '(65, 41)','(86, 40)','(80, 35)', '(91, 35)', '(75, 38)', '(67, 37)']
    def waypointOptimalSeq(self,agent,Items):
        assert self.min_aisle <= int(literal_eval(agent[1])[1]) <= self.max_aisle

        for i in Items:
            assert int(literal_eval(i)[1]) in self.StorageAisles.keys()

        # find waypoint solution
        optimalSeq = [agent]
        items_c = Items.copy()
        start_pt = self.BAYs.eqivalent_loc(loc=agent[1])
        if start_pt in ['n'+str(i) for i in range(1,37)]:   # for example, agentLoc = 'n1'
            start_pt = str(self.eqLocs[start_pt])
        #print('start_pt = ', start_pt)

        while len(items_c) > 0:
            aisle = self.nextAisle(agentLoc=start_pt,items=items_c) # here, aisle = (ret_aisle,side)
            #print('aisle in waypointOptimalSeq = ', aisle)
            cur_aisleItems = self.aisleItemLocs(aisle=aisle,items=items_c)
            #print('cur_aisleItems = ',cur_aisleItems)
            partialSeq = self.aisleItemsOptimalSeq(agentLoc=start_pt,aisleItems=cur_aisleItems)

            optimalSeq = optimalSeq + partialSeq[1:]
            start_pt = optimalSeq[-1]

            #print('cur_aisleItems = ', cur_aisleItems)
            #print('items_c = ', items_c)
            for i in cur_aisleItems:
                #print('i = ', i)
                if int(literal_eval(i)[1]) != 29:
                    items_c.remove(i)
                else: # i looks like "(65, 29, 'N')"
                    #print('i in ailse 29 = ', str((literal_eval(i)[0],literal_eval(i)[1])))
                    items_c.remove(str((literal_eval(i)[0],literal_eval(i)[1])))

        return optimalSeq


    # given items (locations) and agents (locations), find cluster of items assigned to each agent.
    # this is aisle-based cluster-analysis.
    # here, for example,
    # items = ['(53, 40)', '(74, 6)', '(65, 41)',...]
    # n_agents: number of clusters
    def aisle_clusters(self,items=[],n_agents=1):
        ailes = list(set([literal_eval(x)[1] for x in items]))
        if len(ailes) > 1:
            self.aisleDistanceMatrixCreator()
            self.aisleDistanceMatrix_cur = self.aisleDistanceMatrix[ailes].loc[ailes]
            clustering2 = AgglomerativeClustering(n_clusters=n_agents, \
                                        linkage='complete', \
                                        affinity='precomputed', \
                                        connectivity=None, \
                                        compute_full_tree=True, \
                                        distance_threshold=None).fit(self.aisleDistanceMatrix_cur) # "less than" distance_threshold

            idx2 = clustering2.labels_
            #print('idx2 = ', idx2)

            assignments = {}
            for agent in range(0,len(set(idx2))):
                #print(agent)
                assignments[agent] = []

            for i in ailes:
                #print(i,ailes.index(i))
                assignments[idx2[ailes.index(i)]].append(i)

        else:
            # clustering doesn't work if there is only 1 thing to cluster; in this case, 1 aisle
            assignments = {}
            for agent in range(n_agents):
                assignments[agent] = ailes
        return assignments


    # given multi-agents (agents) and items (locations),
    # find Waypoint optimal item sequence for each agent in agentLocs
    # for example, agents = [('agent1', '(86, 42)'), ('agent2', '(51, 43)')]
    #              items = ['(53, 40)', '(74, 6)', '(65, 41)',...]
    # r: 0 < r <= 1; ratio to balance clusters' sizes.
    def waypointOptimalSeq_multiAgents(self,agents,Items,r=0.0):
        print(Items)
        aisle_clusters = self.aisle_clusters(items=Items,n_agents=len(agents))

        if r > 0 and r <=1:
            # cluster balancing
            #print('aisle_clusters = ', aisle_clusters)
            balanced_aisleClusters = self.ailesClusters_balance(clusters=aisle_clusters,ratio=r)
            #print('balanced_aisleClusters = ', balanced_aisleClusters)
            aisle_clusters = balanced_aisleClusters.copy()

        agents_dic = {}
        for a in agents:
            agents_dic[self.BAYs.eqivalent_loc(a[1])] = a

        agents_locs = [self.BAYs.eqivalent_loc(x[1]) for x in agents] # here, x looks like ('agent1', '(86, 42)')
        agents_locs_c = agents_locs.copy()

        ret_mutiAgnets_OptmalSeq = {}

        for k in aisle_clusters.keys():
            #print('#######################################')
            #print('a_cluster = ', aisle_clusters[k])
            cluster_items = [x for x in Items if literal_eval(x)[1] in aisle_clusters[k]]
            #print('cluster_items = ', cluster_items)
            #print('agents_locs_c = ', agents_locs_c)

            self.BAYs.update_cur_DistMatrix(leafnodeNames=list(set(agents_locs_c+cluster_items)))
            ret_agent_loc = self.BAYs.closestLoc_cluster(locs=agents_locs_c,cluster=cluster_items)
            #print('ret_agent_loc = ',ret_agent_loc)
            #print('agent =', agents_dic[ret_agent_loc])

            agents_locs_c.remove(ret_agent_loc)

            ret_singleAgent_OptimalSeq = \
                self.waypointOptimalSeq(agent=agents_dic[ret_agent_loc],Items=cluster_items)
            #print('ret_singleAgent_OptimalSeq = ', ret_singleAgent_OptimalSeq)

            #ret_mutiAgnets_OptmalSeq['task'+str(k+1)] = ret_singleAgent_OptimalSeq
            ret_mutiAgnets_OptmalSeq[k] = ret_singleAgent_OptimalSeq

        return ret_mutiAgnets_OptmalSeq


    ################ Cluster Balancing ###################

    # find the cluster of smallest size
    # cluster: for example,
    #           {1: [0, 3, 11, 16, 17, 19, 20, 21, 22, 23], 2: [1, 7, 12, 18], 0: [2, 4, 5, 6, 8, 9, 10, 13, 14, 15]}
    #     or    {0:['(79, 27)', '(75, 34)', ...],1:['(71, 17)', '(71, 20)', ...],2:['(65, 4)', '(56, 3)', '(54, 7)', '(52, 10)']}
    # return (min_key,clusters[min_key]) where min_key is the key corresponding to the cluster of smallest size in clusters
    def smallestCluster(self,clusters={}):
        if len(clusters) == 0:
            return []
        else:
            min_len = min([len(v) for k, v in clusters.items()])
            min_key = [k for k, v in clusters.items() if len(v) == min_len][0]
            return ((min_key,clusters[min_key]))

    # return (min_distance,minAilses[0],minAilses[1])
    # where min_distance is the minimum distance between minAilses[0] in cluster1 and minAilses[1] in cluster2
    def Cluster_Cluster_dist(self,cluster1,cluster2):
        min_distance = min([abs(c1-c2) for c1 in cluster1 for c2 in cluster2])
        minAilses = [(c1,c2) for c1 in cluster1 for c2 in cluster2 if abs(c1-c2) == min_distance][0]

        return (min_distance,minAilses[0],minAilses[1])

    # return such as (0, (8, 9, 17)) for
    #       clusters = {0: [17, 18, 20, 21, 23, 24, 26, 27, 29, 34, 35, 37, 38, 40, 41], 1: [3, 4, 6, 7, 9]}
    # where 0 is the key witch clusters[0] is closest to clusters[minKey] where minKey != 0 (in this case, minKey = 1)
    #       8 is the distance between aisle 9 in clusters[minKey] and aisle 17 in clusters[0];
    #         i.e. 8 is minimum distance between clusters[minKey] and clusters[0]
    # here, clusters[minKey] is of minimum size out of clusters.
    def nearestCluster(self,minKey,clusters):
        min_distance = min([self.Cluster_Cluster_dist(clusters[minKey],clusters[c])[0] for c in clusters.keys() if c != minKey])
        ret_cluster = [(c,self.Cluster_Cluster_dist(clusters[minKey],clusters[c])) for c in clusters.keys() \
                                                   if self.Cluster_Cluster_dist(clusters[minKey],clusters[c])[0] == min_distance][0]

        return (ret_cluster[0],ret_cluster[1])


    # aisle-clusters (sizes) balancing.
    # here, clusters looks like:
    #             {0: [17, 18, 20, 21, 23, 24, 26, 27, 29, 34, 35, 37, 38, 40, 41], 1: [3, 4, 6, 7, 9]}
    # r: 0 < r <= 1; ratio to balance clusters' sizes. for example, r=1 means all the clusters areof the same sizes
    #                r=0.5 means the sizes across different clusters could differ in two-times, say one is 10 while another is 5.
    def ailesClusters_balance(self,clusters={},ratio=0.01):
        clusters_cp = clusters.copy()
        #print('#### clusters = ', clusters_cp)

        min_cluster = self.smallestCluster(clusters=clusters_cp) # min_cluster likes (1, [3, 4, 6, 7, 9])
        #print('min_cluster = ', min_cluster)

        nearest_cluster = self.nearestCluster(minKey=min_cluster[0],clusters=clusters_cp) # minDist likes (0, (8, 9, 17))
        #print('nearest_cluster = ',nearest_cluster)

        while ratio > len(clusters_cp[min_cluster[0]])/len(clusters_cp[nearest_cluster[0]]):
            clusters_cp[min_cluster[0]].append(nearest_cluster[1][2])
            clusters_cp[nearest_cluster[0]].remove(nearest_cluster[1][2])

            #print('#########')
            min_cluster = self.smallestCluster(clusters=clusters_cp)
            #print('min_cluster = ', min_cluster)

            nearest_cluster = self.nearestCluster(minKey=min_cluster[0],clusters=clusters_cp)
            #print('nearest_cluster = ',nearest_cluster)

        return clusters_cp

    # Similar to BAY2.Optimal_MELDLocs(...) except for
    # convert the sided MELD location such as I212970AAS to non-sided MELD location I212970AA
    # because current Robot can not recoganize side-digit 'N' or 'S'
    # Note: this is only for temorary use.
    def Optimal_MELDLocs(self,MELDlocs,OptimalAssignments):
        ret_OptMELDLocs = self.BAYs.Optimal_MELDLocs(MELDlocs=MELDlocs,OptimalAssignments=OptimalAssignments)

        for k in ret_OptMELDLocs.keys():
            for index, item in enumerate(ret_OptMELDLocs[k]['locations']):
                #print(index, item)
                if len(item) == 10:
                    ret_OptMELDLocs[k]['locations'][index] = str(item)[:9]

        return ret_OptMELDLocs
