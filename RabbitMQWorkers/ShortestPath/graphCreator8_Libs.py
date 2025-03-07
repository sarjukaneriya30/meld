# graphCreator8.py is created based on graphCreator7.py with following new features:
# (1) for aisle 29 in BAY2, agent can access storage location in both northern (N) side and sothern (S) side.
#     (i)   the graph leaf-node class need to have info about side of storage location;
#     (ii)  the graph need to have different 'locations' for both sides;
#     (iii) the optimal sequence algorithms are modified accordingly while shortest path algothm between two
#           storage locations are unaffected.
# (2) Implement general algorithm (see selfOptimalSequence2() and self.)
#      (i)  calculate greedy alg solution as initial solution (without sided nodes included);
#      (ii) Based on initial solution in (i), calculate backtracking solution (without sided nodes included);
#      (iii) With the solution in (ii) as initial solution, recalcualte backtracking solution with all sided nodes added;
#       (iv) improve the solution in (iii) with local best/shortest sequence.

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
#####################################

class Warehouse:
    # Initialize the class. Here nodes is the list of major nodes (the nodes in road intersptions)
    def __init__(self, name='',nodes=[], directed=True,warehouse_database=None,NotReach_locs=[]):
        self.name = name
        self.directed = directed
        self.WH_database = warehouse_database
        self.cur_graph_df = None # current graph (currently-updated graph)
        self.alt_graph_df = pd.DataFrame() # altative graph_df: store modified (such as aisles/edges are blocked) graph_df
        self.n_majorNodes = 0
        self.FullDistMatrix = pd.DataFrame()
        self.cur_DistMatrix = pd.DataFrame()
        self.NotReach_locs = NotReach_locs
        self.cur_NotReach_locs = self.NotReach_locs
        self.TwoSidedLeadNodes = []     # two sided leaf-nodes look like (61,29,'N') and (61,29,'S') in BAY2

        if len(nodes) == 0:
            self.graph_df = pd.DataFrame() # full graph
            self.majorNodes = []

        if len(nodes) > 0:
            self.graph_df = pd.DataFrame(data=np.empty(shape=(len(nodes),len(nodes))),columns=[node.name for node in nodes])
            self.graph_df.index = self.graph_df.columns
            self.graph_df[0:][0:] = np.nan
            np.fill_diagonal(self.graph_df.values, 0)

            self.majorNodes = nodes
            self.n_majorNodes = len(self.majorNodes)

        if not directed:
            self.make_undirected()


    # Add a link from A and B of given distance, and also add the inverse link if the graph is undirected
    def connect(self, A, B, distance=0):
        self.graph_df[B].loc[A] = distance  # B-column; A-row

        if not self.directed:
            self.graph_df[A].loc[B] = distance


    # Create an undirected graph by adding symmetric edges
    def make_undirected(self):
        self.directed = False

        for idx in self.graph_df.index:
            for col in self.graph_df.columns:
                dist = self.graph_df[col].loc[idx]
                self.graph_df[idx].loc[col] = dist

    # add a node
    def add_node(self,node,is_major_node=True):
        if not node.name in self.graph_df.columns:
            # add one more column
            self.graph_df[node.name] = np.nan

            # add one more row
            self.graph_df = \
                self.graph_df.append(pd.DataFrame(data=np.zeros(shape=(1,len(self.graph_df.columns))),\
                                                  columns=self.graph_df.columns,index=[node.name]))

            self.graph_df[:].loc[node.name] = np.nan
            np.fill_diagonal(self.graph_df.values, 0)

            if is_major_node:
                self.majorNodes.append(node)


    # Return a list of nodes in the graph
    def node_names(self):
        return list(self.graph_df.columns)

    # this is an 'abstract' class that will be over-wrriten by child class
    def eqivalent_loc(self,loc=''):
        return loc

    # convert DataFrame graph to scipy.sparse.csgraph
    def converter_OLD1(self,starting_node):
        temp_graph = self.cur_graph_df.copy()

        col_list = list(self.cur_graph_df.columns)
        col_list.remove(starting_node)
        temp_cols = [starting_node] + col_list

        temp_graph = temp_graph[temp_cols]
        #return temp_graph

        csgraph = [list(temp_graph.loc[starting_node])]

        for node in col_list:
            csgraph.append(list(temp_graph.loc[node]))

        return csgraph

    # conver self.cur_graph_df to list-formated graph such as
    # [[1,2,3],[2,3,4],[3,4,5]]
    # here, starting_node is a leaf-node (must not be major-node)
    def converter_OLD2(self,starting_node):
        temp_graph = self.cur_graph_df.copy()

        col_list = list(self.cur_graph_df.columns)
        col_list.remove(starting_node)
        temp_cols = [starting_node] + col_list  # put starting_node at head of temp_cols
        temp_graph = temp_graph[temp_cols].loc[temp_cols]
        #return temp_graph

        temp_graph = temp_graph[temp_cols]

        csgraph = [list(temp_graph.loc[starting_node])]

        for node in col_list:
            csgraph.append(list(temp_graph.loc[node]))

        return csgraph

    # convert self.cur_graph_df to list-formated graph such as
    # [[1,2,3],[2,3,4],[3,4,5]]
    # here, starting_node and end_node can be either leaf-node or major-node
    # Note: end_node is for major node as end_node (not implemented yet).
    def converter(self,starting_node,end_node=''):
        temp_graph = self.cur_graph_df.copy()

        if starting_node in [x.name for x in self.majorNodes]:
            # create a node 'first' that is equivilant to starting_node,
            # i.e. 'first' has the same connections to all other major nodes as starting node has,
            # but not connected to starting_node (a different node fron starting node)
            ret_graph = pd.DataFrame(temp_graph.loc[starting_node]).T
            ret_graph = ret_graph.append(temp_graph)
            ret_graph = pd.DataFrame(ret_graph,columns=['first']+list(ret_graph.columns))
            ret_graph.index = ret_graph.columns
            ret_graph['first'] = ret_graph.loc['first']
            ret_graph['first']['first'] = 0
            ret_graph[starting_node]['first'] = np.nan
            ret_graph['first'][starting_node] = np.nan

            csgraph = []
            for node in ret_graph.columns:
                csgraph.append(list(ret_graph.loc[node]))
        else:
            col_list = list(self.cur_graph_df.columns)
            col_list.remove(starting_node)
            temp_cols = [starting_node] + col_list  # put starting_node at head of temp_cols
            temp_graph = temp_graph[temp_cols].loc[temp_cols]
            #return temp_graph

            temp_graph = temp_graph[temp_cols]
            csgraph = [list(temp_graph.loc[starting_node])]

            for node in col_list:
                csgraph.append(list(temp_graph.loc[node]))

        return csgraph

    # Given a list of locations, (a) find all and return not-reaechable locations;
    #                            (b) return a list of all reachable locations.
    def NotReach_Locs_Processor2(self,list_locs=[]):
        Not_reach = []
        reach = list_locs
        for i in reach:
            if i in self.cur_NotReach_locs:
                Not_reach.append(i)
                reach.remove(i)

        return (Not_reach,reach)


    # (1) Find all not-reachable locations in self.graph_df and self.cur_graph_df
    # (2) update self.NotReach_locs and self.cur_NotReach_locs
    # (3) Remove self.NotReach_locs and self.cur_NotReach_locs from self.graph_df or self.cur_graph_df respectively
    def NotReach_Locs_Processor1(self):
        sumTotal = self.graph_df.sum(axis=0) + self.graph_df.sum(axis=1)
        sumTotal0 = list(sumTotal[sumTotal == 0].index)
        self.NotReach_locs = sumTotal0
        self.remove_nodes(nodesRemoved=self.NotReach_locs,graph_df=True)

        sumTotal = self.cur_graph_df.sum(axis=0) + self.cur_graph_df.sum(axis=1)
        sumTotal0 = list(sumTotal[sumTotal == 0].index)
        self.cur_NotReach_locs = sumTotal0
        self.remove_nodes(nodesRemoved=self.cur_NotReach_locs,graph_df=False)

        return None


    # remove all nodes in nodesRemoved from self.graph_df or self.cur_graph_df
    # Note: this function is unfinished
    def remove_nodes(self,nodesRemoved=[],graph_df=False):
        return

    # create/update current graph (layout strucutre) of majorNodes + leafnodeNames
    def cur_graph(self,leafnodeNames=[],by_full_graph=True):
        ##print('leafnodeNames =', leafnodeNames)
        #all_nodeNames1 = list(set([node.name for node in self.majorNodes] + leafnodeNames))
        majorNdeNames = [node.name for node in self.majorNodes]
        leafnodeNames_c = leafnodeNames.copy()
        for n in leafnodeNames:
            if n in majorNdeNames:
                leafnodeNames_c.remove(n)

        all_nodeNames1 = majorNdeNames + leafnodeNames_c

        ret = self.NotReach_Locs_Processor2(list_locs=all_nodeNames1)
        all_nodeNames = ret[1]

        if by_full_graph: # update self.cur_graph_df with self.graph_dfwith all major-nodes plus leafnodeNames
            self.cur_graph_df = self.graph_df[all_nodeNames].loc[all_nodeNames]
        else: # update self.cur_graph_df with self.alt_graph_df that is modified because aisles are blocked
            if self.alt_graph_df.shape[0] == 0: # if alt_graph_df.shape is empty dataframe: call by self.disconnect(...)
                self.alt_graph_df = self.graph_df[all_nodeNames].loc[all_nodeNames]

            self.cur_graph_df = self.alt_graph_df[all_nodeNames].loc[all_nodeNames]   # call by self.warehouse_shortest_path(...)

    # return a major node of given node name
    def majorNode(self,NodeName):
        #print(NodeName)
        for node in self.majorNodes:
            if node.name == NodeName:
                return node

        return None

    # update self.cur_DistMatrix with with self.FullDistMatrix given leafnodeNames
    # here, leafnodeNames may include major-nodes names such as 'n1', 'n2' etc.
    def update_cur_DistMatrix(self,leafnodeNames=[]):
        reachableNodes = self.NotReach_Locs_Processor2(list_locs=leafnodeNames)
        leafnodeNames1 = reachableNodes[1]
        self.cur_DistMatrix = self.FullDistMatrix[leafnodeNames1].loc[leafnodeNames1]

        return None


    # 1. create self.cur_graph of all major-nodes plus given leafnodeNames
    # 2. disconnect two major nodes;
    # 3. create self.cur_DistMatrix only for all major-nodes plus given leafnodeNames
    def disconnect(self, A, B, leafnodeNames=[]):
        self.cur_DistMatrix = pd.DataFrame(data=np.empty(shape=(len(leafnodeNames),len(leafnodeNames))),columns=[node for node in leafnodeNames])
        self.cur_DistMatrix.index = self.cur_DistMatrix.columns
        self.cur_DistMatrix[0:][0:] = np.nan
        np.fill_diagonal(self.cur_DistMatrix.values, 0)

        # create self.alt_graph of all major-nodes plus given leafnodeNames
        # note: self.alt_graph includes all major-nodes automatically by calling self.cur_graph(...)
        self.alt_graph_df = pd.DataFrame()
        self.cur_graph(leafnodeNames,by_full_graph=False) # update self.cur_graph_df with self.alt_graph_df
                                                          # Both self.cur_graph_df and self.alt_graph_df contain major nodes plus given leafnodeNames

        # disconnect two major nodes A and B:
        self.alt_graph_df[B].loc[A] = np.nan  # B-column; A-row

        if not self.directed:
            self.alt_graph_df[A].loc[B] = np.nan

        # create self.cur_DistMatrix only based on self.cur_graph_df updated with alt_graph_df in self.cur_graph(...)
        untouched = leafnodeNames.copy()
        for node1 in leafnodeNames:
            untouched.remove(node1)
            for node2 in untouched:
                #print(node1, '->', node2)
                TwoLocs = [node1,node2]
                # here, 'by_full_graph=False' instructs that self.cur_graph_df be updated with alt_graph_df in self.cur_graph(...)
                shortestDist, shortestPath = self.warehouse_shortest_path(TwoLocs,by_full_graph=False)
                #print('shortestDist = ',shortestDist)
                self.cur_DistMatrix[node1].loc[node2] = shortestDist
                self.cur_DistMatrix[node2].loc[node1] = shortestDist


    # add all leaf nodes to graph_df
    def add_leafNodes_to_graph(self):
        for node in self.majorNodes:
            #print('*********  ',node.name)
            for leaf_node in node.branch_Nodes_left:
                #print('leaf_node = ',leaf_node)
                self.add_node(leaf_node,is_major_node=False)
                self.connect(node.name,leaf_node.name, distance=leaf_node.cost[0]+node.verticalSpace/2.0)

            for leaf_node1 in node.branch_Nodes_left:
                for leaf_node2 in node.branch_Nodes_left:
                    if leaf_node1.location[1] == leaf_node2.location[1]:
                        self.connect(leaf_node1.name,leaf_node2.name, distance=abs(leaf_node1.cost[0]-leaf_node2.cost[0])+0.001)
                    else:
                        self.connect(leaf_node1.name,leaf_node2.name, distance=abs(leaf_node1.cost[0]-leaf_node2.cost[0])+node.verticalSpace)


            for leaf_node in node.branch_Nodes_right:
                #print('leaf_node = ',leaf_node)
                self.add_node(leaf_node,is_major_node=False)
                self.connect(node.name,leaf_node.name, distance=leaf_node.cost[0]+node.verticalSpace/2.0)

            for leaf_node1 in node.branch_Nodes_right:
                for leaf_node2 in node.branch_Nodes_right:
                    if leaf_node1.location[1] == leaf_node2.location[1]:
                        self.connect(leaf_node1.name,leaf_node2.name, distance=abs(leaf_node1.cost[0]-leaf_node2.cost[0])+0.001)
                    else:
                        self.connect(leaf_node1.name,leaf_node2.name, distance=abs(leaf_node1.cost[0]-leaf_node2.cost[0])+node.verticalSpace)
        return


    # Given major node, create associated leaf-nodes
    def leafNodesCreater(self,cols=[],rows=[],majorNodeName=[],BinLen=0,HalfBlankLen=0,left=True,HalfVlen_aisle=0.0):
        if(len(cols) == 0):
            return []

        leafNodes = []
        minCol = min(cols)
        maxCol = max(cols)
        for col in cols:
            for row in rows:
                ##print('col = ', col, 'raw = ', row)
                if left:
                    cost = HalfBlankLen + (abs(col - minCol) + 0.5)*BinLen + HalfVlen_aisle
                else:
                    cost = HalfBlankLen + (abs(col - maxCol) + 0.5)*BinLen + HalfVlen_aisle

                leafNodes.append(branch_Node(name=str((col,row)), location=(col,row), majorNode=majorNodeName, cost=[cost]))

        return leafNodes

    # Given major node, create associated sided-leaf-nodes
    # side: 'N': Northern side of the sided-leaf-node; 'S': Sorthern side of sided-leaf-node;
    #        '': only one side (normal leaf-node) which is handled by self.leafNodesCreater(...)
    def leafNodesCreater2(self,cols=[],rows=[],majorNodeName=[],BinLen=0,HalfBlankLen=0,left=True,HalfVlen_aisle=0.0,side=''):
        if(len(cols) == 0):
            return []

        leafNodes = []
        minCol = min(cols)
        maxCol = max(cols)
        for col in cols:
            for row in rows:
                ##print('col = ', col, 'raw = ', row)
                if left:
                    cost = HalfBlankLen + (abs(col - minCol) + 0.5)*BinLen + HalfVlen_aisle
                else:
                    cost = HalfBlankLen + (abs(col - maxCol) + 0.5)*BinLen + HalfVlen_aisle

                leafNodes.append(branch_Node(name=str((col,row,side)), location=(col,row), majorNode=majorNodeName, cost=[cost],side=side))

        return leafNodes

    # Create shortest pick path from predecessors
    def shortest_path_creator(self,predecessors = []):
        #print(predecessors)
        ret_shortest_path = ['target']
        cur_node = predecessors[-1]
        while not cur_node == -9999:
            pre_node = predecessors[cur_node]
            ret_shortest_path.append(cur_node)
            #print(cur_node, pre_node)
            #print(ret_shortest_path)
            cur_node = pre_node

        ret_shortest_path.reverse()  # store reverse of ret_shortest_path into ret_shortest_path
        return ret_shortest_path

    # Create shortest path between two given warehouse locations (self.cur_graph_df)
    def warehouse_shortest_path(self,TwoLocs=[],by_full_graph=True):
        self.cur_graph(TwoLocs,by_full_graph)

        converted1 = self.converter(TwoLocs[0])
        ##print('converted1 = ',converted1)
        csrMatrix = csr_matrix(converted1)

        # indices=0 means that node 0 is the starting point
        dist_matrix, predecessors = shortest_path(csgraph=csrMatrix, method='D',directed=False, indices=0, return_predecessors=True)
        #print('predecessors = ', predecessors)

        return (dist_matrix[-1],self.shortest_path_creator(predecessors))

    # Create shortest path given a list of warehouse locations (self.cur_graph_df)
    def warehouse_shortest_path2(self,ItemLocations=[],by_full_graph=True):
        #print('ItemLocations = ', ItemLocations)
        shortest_path = []

        start_pt = ItemLocations[0]
        shortest_path.append(start_pt)
        start_pt = self.eqivalent_loc(start_pt)
        for loc in ItemLocations[1:]:
            TwoLocs = [start_pt,loc]
            ##print('TwoLocs = ', TwoLocs)
            ret = self.warehouse_shortest_path(TwoLocs,by_full_graph)
            start_pt = loc

            #print(ret)
            if len(ret[1]) > 2:
                for i in ret[1][1:-1]:
                    shortest_path.append(i)

            shortest_path.append(start_pt)

        return shortest_path

    # Create shortest path given several lists of warehouse locations (multi-agents)
    def warehouse_shortest_path3(self,MultiItemListsLocs={},by_full_graph=True):
        Multi_shortest_path = {}
        for k in MultiItemListsLocs.keys():
            #print('k = ', k)
            itemLocs = MultiItemListsLocs[k].copy()
            itemLocs[0] = itemLocs[0][1]
            Multi_shortest_path[k] = self.warehouse_shortest_path2(itemLocs,by_full_graph=True)
            Multi_shortest_path[k][0] = MultiItemListsLocs[k][0]

        return Multi_shortest_path


    # create full (all leafNodes) distance matrix from self.graph_df
    def DistanceMatrix(self):
        allNodes = self.node_names()
        leafNodes = self.node_names()[self.n_majorNodes:]
        self.FullDistMatrix = pd.DataFrame(data=np.empty(shape=(len(allNodes),len(allNodes))),columns=[node for node in allNodes])
        self.FullDistMatrix.index = self.FullDistMatrix.columns
        self.FullDistMatrix[0:][0:] = np.nan
        np.fill_diagonal(self.FullDistMatrix.values, 0)
        untouched = leafNodes.copy()
        #for leaf_node in leafNodes[:1]:
        for leaf_node in leafNodes:
            untouched.remove(leaf_node)
            for leaf_node2 in untouched:
                #print(leaf_node, '->', leaf_node2)
                ##print('type = ', type(leaf_node))
                TwoLocs = [leaf_node,leaf_node2]
                shortestDist, shortestPath = self.warehouse_shortest_path(TwoLocs)
                #print('shortestDist = ',shortestDist)
                self.FullDistMatrix[leaf_node].loc[leaf_node2] = shortestDist
                self.FullDistMatrix[leaf_node2].loc[leaf_node] = shortestDist

        self.cur_DistMatrix = self.FullDistMatrix.copy()  # initialize self.cur_DistMatrix

    # add all major nodes to self.FullDistMatrix created by DistanceMatrix(...)
    # call DistanceMatrix(...) first and then DistanceMatrixMajorNodes(...)
    def DistanceMatrixMajorNodes(self):
        # create distance matrix across all major nodes
        MajorNodes_c = [node.name for node in self.majorNodes]
        untouched = MajorNodes_c.copy()
        for node1 in MajorNodes_c:
            untouched.remove(node1)
            for node2 in untouched:
                #print(node1, '->', node2)
                TwoLocs = [node1,node2]
                converted1 = self.converter_MajorNodes(TwoLocs=TwoLocs)
                csrMatrix = csr_matrix(converted1)

                # indices=0 means that node 0 is the starting point
                dist_matrix, predecessors = shortest_path(csgraph=csrMatrix, method='D',directed=False, indices=0, return_predecessors=True)
                #print('shortestDist = ',dist_matrix[-1])
                self.FullDistMatrix[node1].loc[node2] = dist_matrix[-1]
                self.FullDistMatrix[node2].loc[node1] = dist_matrix[-1]

        # create distance matrix between all major nodes and all leaf nodes
        LeafNodes_c = self.node_names()[self.n_majorNodes:]
        for MajorNode in MajorNodes_c:
            for LeafNode in LeafNodes_c:
                #print(MajorNode, '->', LeafNode)
                TwoLocs = [MajorNode,LeafNode]
                converted1 = self.converter_MajorLeafNodes(TwoLocs=TwoLocs)
                csrMatrix = csr_matrix(converted1)

                # indices=0 means that node 0 is the starting point
                dist_matrix, predecessors = shortest_path(csgraph=csrMatrix, method='D',directed=False, indices=0, return_predecessors=True)
                #print('shortestDist = ',dist_matrix[-1])
                self.FullDistMatrix[MajorNode].loc[LeafNode] = dist_matrix[-1]
                self.FullDistMatrix[LeafNode].loc[MajorNode] = dist_matrix[-1]

        return None



    # similar to converter(...) except that TwoLocs = ['MajorNode1','MajorMode2']
    # NOTE1: this is only for distance matrix (length of shortest path), not for shortest path.
    # NOTE2: temp_cols contains the major-nodes in a differenrt order from
    #       those in self.cur_graph() which is ['n1','n2','n3',...], therefore, based on
    #       temp_cols,the shortest path may not be correct, but its distance/len is correct.
    def converter_MajorNodes(self,TwoLocs=[]):
        self.cur_graph() # create self.cur_graph_df with all major nodes
        temp_graph = self.cur_graph_df.copy()

        col_list = list(temp_graph.columns)
        col_list.remove(TwoLocs[0])
        col_list.remove(TwoLocs[1])
        temp_cols = [TwoLocs[0]] + col_list + [TwoLocs[1]] # put TwoLocs[0] at head of temp_cols
                                                           # put TwoLocs[1] at the end of temp_cols
        temp_graph = temp_graph[temp_cols].loc[temp_cols]

        csgraph = []
        for node in temp_cols:
            csgraph.append(list(temp_graph.loc[node]))

        return csgraph

    # similar to converter(...) except that TwoLocs = ['MajorNode','LeafNode']
    # NOTE1: this is only for distance matrix (length of shortest path), not for shortest path.
    # NOTE2: temp_cols contains the major-nodes in a differenrt order from
    #       those in self.cur_graph() which is ['n1','n2','n3',...], therefore, based on
    #       temp_cols,the shortest path may not be correct, but its distance/len is correct.
    def converter_MajorLeafNodes(self,TwoLocs=[]):
        self.cur_graph(TwoLocs[1:2]) # create self.cur_graph_df with all major nodes plus 'LeafNode'
        temp_graph = self.cur_graph_df.copy()

        col_list = list(temp_graph.columns)
        col_list.remove(TwoLocs[0]) # remove 'MajorNode'
        temp_cols = [TwoLocs[0]] + col_list

        temp_graph = temp_graph[temp_cols].loc[temp_cols]

        csgraph = []
        for node in temp_cols:
            csgraph.append(list(temp_graph.loc[node]))

        return csgraph

    #############################
    # shortest/optimal sequence

    # convert leaf-nodes of two sides (such those leaf-nodes in BAY2)
    # to their sided nodes such as (67,29,'N'), (67,29,'S').
    def SidedLeafNodes_converter(self,items=''):
        ret_items = []
        for i in items:
            if i in [n.name for n in self.majorNodes]:
                ret_items.append(i)
            else:
                if i in self.TwoSidedLeadNodes: # such as (67,29,'N')
                    ret_items.append(str((literal_eval(i)[0],literal_eval(i)[1],'N')))
                    ret_items.append(str((literal_eval(i)[0],literal_eval(i)[1],'S')))
                else:
                    ret_items.append(i)

        return ret_items

    # calculate path (total) distance
    def path_dist(self,start_pt,a_path):
        re_dist = self.cur_DistMatrix[start_pt][a_path[0]]
        for i in range(0,len(a_path)-1):
            re_dist = re_dist + self.cur_DistMatrix[a_path[i]][a_path[i+1]]

        return re_dist

    ####### All permutations: shortest sequence
    def ShortestItemSequence(self,items,start_pt):
        perm = permutations(items)
        perm_list = list(perm)

        ret_path = list(perm_list[0])
        ret_path.insert(0,start_pt)
        min_d = self.path_dist(start_pt,ret_path)
        for a_path in perm_list:
            d = self.path_dist(start_pt,a_path)
            ##print(a_path,d)
            if min_d > d:
                ret_path = list(a_path)
                ret_path.insert(0,start_pt)
                min_d = d

        return ret_path

    ########################################
    # modified based on ShortestItemSequence, add end_pt
    def ShortestItemSequence2(self,items,start_pt,end_pt):
        perm = permutations(items)
        perm_list = list(perm)

        ret_path = list(perm_list[0])
        ret_path.insert(0,start_pt)
        ret_path.append(end_pt)
        min_d = self.path_dist(ret_path[0],ret_path[1:])
        for a in perm_list[1:]:
            a_path = list(a)
            a_path.append(end_pt)
            d = self.path_dist(start_pt,a_path)
            ##print(a_path,d)
            if min_d > d:
                ret_path = list(a_path)
                ret_path.insert(0,start_pt)
                min_d = d

        return ret_path

#################### greedy algorithm
    def greedy(self,start_pt,items):
        items_cp = items.copy()
        ##print([start_pt] + items_cp)

        ret_path = [start_pt]
        cur_pt = start_pt
        for pi in items:
            if len(items_cp) > 0:
                min_pt = self.greedyPt(cur_pt,items_cp)
                ##print('###############')
                ##print('ret_path: ',ret_path)
                ##print('cur_pt: ',cur_pt)
                ##print('items_cp: ',items_cp)
                ##print('min_pt: ',min_pt)
                ##print('len(items_cp): ',len(items_cp))

                if not min_pt in ret_path:
                    ret_path.append(min_pt)

                cur_pt = min_pt
                ##print('###############')
                ##print('min_pt: ',min_pt)
                if len(literal_eval(min_pt)) == 3:    # storage location of two sides, such as those in aisle 29 in BAY2
                    ##print('min_pt in 111 =',literal_eval(min_pt))
                    items_cp.remove(min_pt)
                    if literal_eval(min_pt)[2].strip() == 'N':
                        min_pt2 = str((literal_eval(min_pt)[0],literal_eval(min_pt)[1],'S'))
                    else:
                        min_pt2 = str((literal_eval(min_pt)[0],literal_eval(min_pt)[1],'N'))

                    ##print('min_pt2 = ',min_pt2)
                    if min_pt2 in items_cp:
                        items_cp.remove(min_pt2)
                else:
                    items_cp.remove(min_pt)

        return ret_path


    def greedyPt(self,start_pt,items):
        ret_pt = items[0]

        d = self.cur_DistMatrix[start_pt][items[0]]
        ##print('d = ', d)
        for pt in items:
            if d > self.cur_DistMatrix[start_pt][pt]:
                ret_pt = pt
                d = self.cur_DistMatrix[start_pt][pt]

        return ret_pt

    ########################################
    # modified based on backtrack(...).
    # here, r>=0 is the gauge to control 'cut off subtrees': default: 0; larger value means more subtrees
    # are kept for searching (longer CPU time). if r is plus infinity, then backtrack2 is actually to look
    # for best solution.
    def backtrack2_OriginalBAK(self,start_pt,items,r=0,ret_items=[],min_path={},depth=0,min_d=[9999999]):
        if depth == 0:   # in the start_pt level
            ret_items=[]
            min_path={}
            min_d=[9999999]

        depth = depth + 1

    #    #print('####################')
    #    #print('depth in 111 = ',depth)
    #    #print('items in 111 = ',items)
    #    #print('min_d[-1] in 111 = ',min_d[-1])
        it_count = -1
        for i in items:
            it_count = it_count + 1

            if len(ret_items) == 0: ret_items = [start_pt]

            if it_count > 0:  # imply len(ret_items) > 1
                ret_items.remove(ret_items[-1])  # remove the last item added in previous backtrack call

        #        #print('start_pt, i, in 222 = ',start_pt, '  ',i)
            ##print('ret_items in 222 = ',ret_items)

            ##print('it_count in 111 = ',it_count)
            ##print('i in 111 = ',i)
            ##print('ret_items in 111 = ',ret_items)
            if not i in ret_items:
                ret_items = ret_items + [i]  # add in item i in current backtrack call
                ##print('ret_items in 222 = ',ret_items)

            d_sum = 0  # length of current ret_items
            for v in range(0,len(ret_items)-1):
                d_sum = d_sum + self.cur_DistMatrix[ret_items[v]][ret_items[v+1]]

            ##print('d_sum in 222 = ', d_sum)

            start_pt = i
            items_cp = items.copy()
            items_cp.remove(i)

            ##print('i in 333 =',i,'  items = ',items)
            ##print('ret_items in 333 = ',ret_items)
            items_cp2 = items_cp.copy()
            gd_dist = 0
            ##print('i in 333 =',i,'  items_cp2 = ',items_cp2)
            if len(items_cp2) > 0:
                ##print('start_pt in 333 =',start_pt,'  items_cp2 = ',items_cp2)
                #ret_gd = greedy(start_pt,[start_pt]+items_cp2,dist_matrix)
                ret_gd = self.greedy(start_pt,items_cp2)
                #            #print('ret_gd in 333 = ', ret_gd)

                if len(ret_gd[1:]) > 0:
                    gd_dist = self.path_dist(ret_gd[0],ret_gd[1:])
                    ##print('gd_dist in 333 = ', gd_dist)

            ##print('Thresholds in 444 = ', min_d[-1], d_sum, gd_dist)
            #min_p = {}
            if min_d[-1] > d_sum + gd_dist - r:
                min_p = self.backtrack2(start_pt,items_cp,r,ret_items,min_path,depth,min_d)
                ##print('min_p in 555 = ',min_p)
                keylist = list(min_p.keys())
                keylist.sort()
                min_d.append(keylist[0])
                ##print('keylist in 555 = ', min_d)
                ##print('min_p in 555 = ', min_p)

            ##print('depth in 555 = ', depth)
            ##print('min_items in 555 = ', min_p)
            ##print('items in 555 = ', items)

        if len(items) == 0:  # at the tree bottom
#           #print('depth in 000 = ',depth)
#           #print('ret_items in 000 = ',ret_items)

            d_sum = 0
            for v in range(0,len(ret_items)-1):
                d_sum = d_sum + self.cur_DistMatrix[ret_items[v]][ret_items[v+1]]

            min_path[d_sum] = ret_items.copy()

            #depth = depth - 1
            return min_path

        return min_path

    ########################################
    # modified based on backtrack(...).
    # here, r>=0 is the gauge to control 'cut off subtrees': default: 0; larger value means more subtrees
    # are kept for searching (longer CPU time). if r is plus infinity, then backtrack2 is actually to look
    # for best solution.
    def backtrack2(self,start_pt,items,r=0,ret_items=[],min_path={},depth=0,min_d=[9999999]):
        if depth == 0:   # in the start_pt level
            ret_items=[]
            min_path={}
            min_d=[9999999]

        depth = depth + 1

        ##print('####################')
        ##print('depth in 111 = ',depth)
        ##print('items in 111 = ',items)
        ##print('min_d[-1] in 111 = ',min_d[-1])
        ##print('min_path in 111 = ', min_path)
        it_count = -1
        for i in items:
            it_count = it_count + 1

            if len(ret_items) == 0: ret_items = [start_pt]

            #pre_i = ret_items[-1] # the last item added in previous backtrack call

            if it_count > 0:  # imply len(ret_items) > 1
                ret_items.remove(ret_items[-1])  # remove the last item added in previous backtrack call

        #        #print('start_pt, i, in 222 = ',start_pt, '  ',i)
            ##print('ret_items in 222 = ',ret_items)

            ##print('it_count in 111 = ',it_count)
            ##print('i in 111 = ',i)
            ##print('ret_items in 111 = ',ret_items)
            if not i in ret_items:
                ret_items = ret_items + [i]  # add in item i in current backtrack call
                ##print('ret_items in 222 = ',ret_items)

            d_sum = 0  # length of current ret_items
            for v in range(0,len(ret_items)-1):
                d_sum = d_sum + self.cur_DistMatrix[ret_items[v]][ret_items[v+1]]

            ##print('d_sum in 222 = ', d_sum)

            ##print('i = ',i)
            start_pt = i
            items_cp = items.copy()
            #items_cp.remove(i)
            if len(literal_eval(i)) == 3:    # storage location of two sides, such as (80, 29, 'N') in aisle 29 in BAY2
                ##print('i in 111 =',literal_eval(i))
                items_cp.remove(i)
                if literal_eval(i)[2].strip() == 'N':
                    i2 = str((literal_eval(i)[0],literal_eval(i)[1],'S'))
                else:
                    i2 = str((literal_eval(i)[0],literal_eval(i)[1],'N'))

                ##print('i2 = ',i2)
                items_cp.remove(i2)
                #if i2 in items_cp:
                #    items_cp.remove(i2)
            else:
                items_cp.remove(i)

            #if len(literal_eval(pre_i)) == 3:    # if the item added in previous backtrack call is a storage location of two sides.
            #     if literal_eval(pre_i)[2].strip() == 'N':
            #        pre_i2 = str((literal_eval(pre_i)[0],literal_eval(pre_i)[1],'S'))
            #     else:
            #        pre_i2 = str((literal_eval(pre_i)[0],literal_eval(pre_i)[1],'N'))

            #     if not pre_i2 in items_cp:
            #        items_cp.append(pre_i2)

            #     #print('i = ',i, '   pre_i = ',pre_i, '  pre_i2 = ',pre_i2)
            #     #print('items_cp = ', items_cp)

            ##print('i in 333 =',i,'  items = ',items)
            ##print('ret_items in 333 = ',ret_items)
            items_cp2 = items_cp.copy()
            gd_dist = 0
            ##print('i in 333 =',i,'  items_cp2 = ',items_cp2)
            if len(items_cp2) > 0:
                ##print('start_pt in 333 =',start_pt,'  items_cp2 = ',items_cp2)
                #ret_gd = greedy(start_pt,[start_pt]+items_cp2,dist_matrix)
                ret_gd = self.greedy(start_pt,items_cp2)
                #            #print('ret_gd in 333 = ', ret_gd)

                if len(ret_gd[1:]) > 0:
                    gd_dist = self.path_dist(ret_gd[0],ret_gd[1:])
                    ##print('gd_dist in 333 = ', gd_dist)

            ##print('Thresholds in 444 = ', min_d[-1], d_sum, gd_dist)
            #min_p = {}
            if min_d[-1] > d_sum + gd_dist - r:
                min_p = self.backtrack2(start_pt,items_cp,r,ret_items,min_path,depth,min_d)
                ##print('min_p in 555 = ',min_p)
                keylist = list(min_p.keys())
                keylist.sort()
                min_d.append(keylist[0])
                ##print('keylist in 555 = ', min_d)
                ##print('min_p in 555 = ', min_p)

            ##print('depth in 555 = ', depth)
            ##print('min_items in 555 = ', min_p)
            ##print('items in 555 = ', items)

        if len(items) == 0:  # at the tree bottom
            #print('depth in 000 = ',depth)
            ##print('ret_items in 000 = ',ret_items)

            d_sum = 0
            for v in range(0,len(ret_items)-1):
                d_sum = d_sum + self.cur_DistMatrix[ret_items[v]][ret_items[v+1]]

            min_path[d_sum] = ret_items.copy()
            #print('s_sum in -111 = ',d_sum)

            #depth = depth - 1
            return min_path

        return min_path

    # Based on backtrack2(...) to add random.shuffle(items)
    # n: number of shuffles. 1 for defult
    # r: threshlod of backtrackig algorithm
    # Note: testing shows that r is more effective than n
    def backtrack3(self,start_pt,items,n=1,r=0):
        items_cp = items.copy()
        ret_bk2 = self.backtrack2(start_pt,items_cp,r)
        mink3 = min(ret_bk2.keys())
        minSeq3 = ret_bk2[mink3]
        #print('**** mink3 = ',mink3)

        for i in range(1,n): # here, range(1,n): 1,2,...,n-1
            if i == 1:
                items_cp.reverse()
            else:
                random.shuffle(items_cp)

            ret_bk2 = self.backtrack2(start_pt,items_cp,r)
            k = min(ret_bk2.keys())
            Seq = ret_bk2[k]

            if k < mink3:
                mink3 = k
                minSeq3 = Seq.copy()

            ##print('**** mink3 = ',mink3)
            ##print(minSeq3)

        return mink3, minSeq3

    # combine self.backtrack3 with bk_improvement2 and bk_improvement3
    # items: locations (col,aisle) of items
    def OptimalSequence(self,start_pt,items,n=1,r=0,d=5):
        #print('nv=v',n,'  r = ',r,  ' d = ',d)

        eq_loc = self.eqivalent_loc(start_pt)  # replace agent location with its eqivalent location
        self.update_cur_DistMatrix(leafnodeNames=list(set([eq_loc]+items)))

        # calculate initial (greedy) solution
        ret_gd = self.greedy(eq_loc,items)

        # based on ret_gd, calcualte (initial) backtracking solution without sided-locations
#NK        ret_bk3 = self.backtrack3(start_pt,items,n,r)
        ret_bk3 = self.backtrack3(eq_loc,ret_gd[1:],n,r)
        ret_bk3[1][0] = start_pt # restore starting point with start_pt

        # based on ret_bk3, calculate final backtracking solution with sided-locations
        items_sided = self.SidedLeafNodes_converter(items=ret_bk3[1][1:])
        self.update_cur_DistMatrix(leafnodeNames=list(set([eq_loc]+items_sided)))
        ret_bk3_sided = self.backtrack3(eq_loc,items_sided,n,r)
        ret_bk3_sided[1][0] = start_pt # restore starting point with start_pt

        return ret_bk3_sided

    # The sme as OptimalSequence(...) except for r1 and r2 instead of r.
    # combine self.backtrack3 with bk_improvement2 and bk_improvement3
    # items: locations (col,aisle) of items
    # here, r1: threshold for initial (without sided-locations) backtracking solution;
    #       r2: threshold for final (with sided-locations) backtracking solution
    def OptimalSequence2(self,start_pt,items,n=1,r1=0,r2=0,d=5):
        #print('n=',n,'  r1=',r1,'  r2=',r2,  ' d=',d)

        eq_loc = self.eqivalent_loc(start_pt)  # replace agent location with its eqivalent location
        self.update_cur_DistMatrix(leafnodeNames=list(set([eq_loc]+items)))

        # step1: calculate initial (greedy) solution
        ret_gd = self.greedy(eq_loc,items)

        # step2: based on ret_gd in step1, calcualte (initial) backtracking solution without sided-locations
#NK        ret_bk3 = self.backtrack3(start_pt,items,n,r)
        ret_bk3 = self.backtrack3(eq_loc,ret_gd[1:],n,r1)
        ret_bk3[1][0] = start_pt # restore starting point with start_pt

        # step3: based on ret_bk3 in step2, calculate final backtracking solution with sided-locations
        items_sided = self.SidedLeafNodes_converter(items=ret_bk3[1][1:])
        self.update_cur_DistMatrix(leafnodeNames=list(set([eq_loc]+items_sided)))
        ret_bk3_sided = self.backtrack3(eq_loc,items_sided,n,r2)
        ret_bk3_sided[1][0] = start_pt # restore starting point with start_pt

        return ret_bk3_sided
        #return ret_bk3

#K        bk_improved2 = bk_improvement2(ret_bk3[1],dist_matrix,d)  # ret_bk3[1] is the optimal sequence returned by backtrack3(...)
#K        bk_improved3 = bk_improvement3(bk_improved2,dist_matrix,d)

#K        ret_sequence = bk_improved3[1]
#K        ret_d = path_dist(ret_sequence[0],ret_sequence[1:],dist_matrix)

#K        bk_improved3 = bk_improvement3(ret_bk3[1],dist_matrix,d)
#K        bk_improved2 = bk_improvement2(bk_improved3[1],dist_matrix,d)

    #    #print('ret_sequence = ', ret_sequence)
    #    #print('bk_improved2 = ', bk_improved2)
#K        d1 = path_dist(ret_sequence[0],ret_sequence[1:],dist_matrix)
#K        d2 = path_dist(bk_improved2[0],bk_improved2[1:],dist_matrix)

#K        if d1 > d2:
#K            ret_sequence = bk_improved2
#K            ret_d = d2

#K        return (ret_d, ret_sequence)



    # For a given initial optimal sequence (initial_seq), OptimalSeqWithLocalShortestSeq creates improved optimal
    # sequence with replacing each section of length sectLen with its corresponding shortest sequence
    # created by calling ShortestItemSequence2(...).
    # here, initial_seq looks like [('(86, 41)', '(78, 41)', '(65, 41)',...]
    def OptimalSeqWithLocalShortestSeq(self,initial_seq,sectLen=6):
        initial_seq_c = initial_seq.copy()
        ret_seq = []

        self.update_cur_DistMatrix(initial_seq_c)

        if sectLen <= 1: return initial_seq

        for i in range(0,len(initial_seq_c)-sectLen):
            ##print('####### i = ', i, '##########')
            ##print(initial_seq_c[i],initial_seq_c[i+1:i+sectLen],initial_seq_c[i+sectLen])
            shortest_sect = self.ShortestItemSequence2(initial_seq_c[i+1:i+sectLen],initial_seq_c[i],initial_seq_c[i+sectLen])
            ##print('shortest_sect = ',shortest_sect)

            #d1 = self.path_dist(start_pt=initial_seq_c[i],a_path=initial_seq_c[i+1:i+sectLen+1])
            #d2 = self.path_dist(start_pt=shortest_sect[0],a_path=shortest_sect[1:])
            ##print('d1 = ', d1, '    d2 = ', d2)

            # update initial_seq_c and ret_seq
            #initial_seq_c = initial_seq_c[:i+1].copy() + shortest_sect.copy() + initial_seq_c[?:].copy()

            if i == 0:
                initial_seq_c = shortest_sect.copy() + initial_seq_c[sectLen+1:].copy()
                ret_seq = shortest_sect
            else:
                initial_seq_c = initial_seq_c[:i].copy() + shortest_sect.copy() + initial_seq_c[i+sectLen+1:].copy()
                ret_seq = ret_seq[:i] + shortest_sect

            ##print('initial_seq_c = ', initial_seq_c)
            ##print('ret_seq = ', ret_seq)

        return ret_seq


    # Based on OptimalSeqWithLocalShortestSeq(...), (i) ieterate sectLen (say, from 4 to 8) and
    # (ii) repeatly calling OptimalSeqWithLocalShortestSeq(...) to achieve final optimal solution
    # here: initial_seq lokks like [('agent1', '(86, 42)'), '(78, 41)', '(65, 41)',...]
    def OptimalSeqWithLocalShortestSeq2(self,initial_seq):
        initial_seq_c = initial_seq.copy()
        initial_seq_c[0] = self.eqivalent_loc(initial_seq[0][1]) # here, initial_seq[0] like ('agent1','(86,42)')

        self.update_cur_DistMatrix(initial_seq_c)

        # setup length of shortest sequence
        min_sectLen = 5
        max_sectLen = 8

        min_seq = initial_seq_c.copy()
        d_min = self.path_dist(min_seq[0],min_seq[1:])

        min_seq_c = min_seq.copy()
        for l in range(min_sectLen,max_sectLen+1):
            ret_seq = self.OptimalSeqWithLocalShortestSeq(initial_seq=min_seq_c.copy(),sectLen=l)
            d = self.path_dist(ret_seq[0],ret_seq[1:])

            if d_min > d:
                min_seq = ret_seq.copy()
                d_min = d

        d_min_pre = decimal.MAX_PREC
        min_seq_cur = min_seq.copy()
        d_min_cur = d_min
        while d_min_cur < d_min_pre:
            #print('*************************')
            #print('d_min_pre, d_min_cur in 111 = ', d_min_pre, d_min_cur)
            d_min_pre = d_min_cur

            min_seq_cur_c = min_seq_cur.copy()
            for l in range(min_sectLen,max_sectLen+1):
                ret_seq = self.OptimalSeqWithLocalShortestSeq(initial_seq=min_seq_cur_c.copy(),sectLen=l)
                d = self.path_dist(ret_seq[0],ret_seq[1:])

                #print('sectLen, d = ', l, d)
                if d_min_cur > d:
                    min_seq_cur = ret_seq.copy()
                    d_min_cur = d

        min_seq_cur[0] = initial_seq[0] # here, initial_seq[0] like ('agent1','(86,42)')

        return (d_min_cur,min_seq_cur)

    # based on OptimalSeqWithLocalShortestSeq2(...), claculate final optimal solution for
    # each of multiple initial solutions initial_seqs where initial_seqs looks like
    # {[('agent1', '(86, 42)'), '(78, 41)', '(65, 41)',...], [('agent2', '(86, 7)'), '(78, 18)', '(75, 41)',...],...}
    def OptimalSeqWithLocalShortestSeq3(self,initial_seqs):
        ret_finalSolutions = {}

        for k in initial_seqs.keys():
            ini_seq = initial_seqs[k].copy()
            ret_finalSolutions['Task'+str(k+1)] = self.OptimalSeqWithLocalShortestSeq2(ini_seq)

        return ret_finalSolutions



    # AgentsAssignment1_1: Multi-Agents items assignments based on backtrack
    #                      solution (bk_minSeq) with engents capacities.
    # AgentsAssignment1_1 does not include mult-agent optimal sequence locations.
    # Here, NIINs: item NIINs of those items in bk_minSeq -- must in the same order;
    #       agents: agent IDs for picking up items.
    #       bk_minSeq: does not include starting point

#    def AgentsAssignment1_1(self,bk_minSeq,item_vols,item_weights):
    def AgentsAssignment1_1(self,bk_minSeq,NIINs,agentIDs):
        assignments = {}
        agentIndex = 0
        assignments[agentIDs[agentIndex]] = []

        AMR_Vol = self.WH_database.agents[agentIDs[agentIndex]].capacity_volumn
        AMR_wt = self.WH_database.agents[agentIDs[agentIndex]].capacity_weight
        agent_vol = 0
        agent_wt = 0

        #print('AMR_Vol = ', AMR_Vol, '  AMR_wt = ', AMR_wt)
        for idx in range(0,len(bk_minSeq)):
            #print('idx = ', idx, '   item = ',bk_minSeq[idx])
            #print('agentIndex = ', agentIndex, '   agent = ', agentIDs[agentIndex])

            item_l = self.WH_database.items[NIINs[idx]].LENGTH
            item_w = self.WH_database.items[NIINs[idx]].WIDTH
            item_h = self.WH_database.items[NIINs[idx]].HEIGHT
            weight = self.WH_database.items[NIINs[idx]].WEIGHT

            if agent_vol + item_l*item_w*item_h < AMR_Vol and agent_wt + weight < AMR_wt:
                assignments[agentIDs[agentIndex]].append(bk_minSeq[idx])
                #assignments[agentIDs[agentIndex]].append(NIINs[idx])

                #print('vol in 111 = ',agent_vol)

                agent_vol = agent_vol + item_l*item_w*item_h
                agent_wt = agent_wt + weight
            else:
                agentIndex = agentIndex + 1
                assignments[agentIDs[agentIndex]] = []

                assignments[agentIDs[agentIndex]].append(bk_minSeq[idx])
                #assignments[agentIDs[agentIndex]].append(NIINs[idx])

                #print('vol in 222 = ',agent_vol)

                agent_vol = item_l*item_w*item_h
                agent_wt = weight

        return assignments

    # Multi-agent (items) assitnments based on backtracking solution:
    # combine self.OptimalSequence(...) with self.AgentsAssignment1_1(...)
    # Criterion (assignment criterion):
    #       1 -- (default) assigned by agent capacity interms of agents' volumns and weights;
    #       2 -- assigned by equal number of items (for nightly scan)
    def AgentsAssignment1_2(self,start_pt,NIINs,agentIDs,Criterion=1,n=1,r=0,d=5):
        # extract items by NIINs
        items = []
        for n1 in NIINs:
            #print('n1 = ',n1)
            col = self.WH_database.items[n1].column
            aisle =  self.WH_database.items[n1].aisle
            #print(col, aisle)
            items.append(str(tuple([col,aisle])))

        #print('items = ',items)

        # calculate optimal sequence for multi-agents assignments
        ret_OptimalSequence = self.OptimalSequence(start_pt,items,n,r,d)
        #print('ret_OptimalSequence = ', ret_OptimalSequence)

        if Criterion == 1:
            # calculate assignments based on optimal sequence by self.OptimalSequence(...)
            ret = self.AgentsAssignment1_1(ret_OptimalSequence[1][1:],NIINs,agentIDs)
        elif Criterion == 2: # calculate assignments based on equal umber of items for each of available agents
            ret = {}
            for agent in agentIDs:
                ret[agent] = []

            AssignedItems = ret_OptimalSequence[1].copy()
            AssignedItems = AssignedItems[1:]

            #print('AssignedItems = ', AssignedItems)
            #print('agentIDs = ',agentIDs)
            n_assign = int(len(AssignedItems)/len(agentIDs))
            if n_assign * len(agentIDs) < len(AssignedItems):
                n_assign = n_assign + 1

            for agentID in agentIDs:
                #print('agentID = ',agentID)
                if len(AssignedItems) >= n_assign:
                    ##print('len of AssignedItems = ', len(AssignedItems))
                    ##print('AssignedItems = ', AssignedItems)
                    ##print('n_assign = ', n_assign)
                    for ii in range(0,n_assign):
                        ##print('ii = ', ii)
                        ret[agentID].append(AssignedItems[ii])

                    for it in ret[agentID]:
                        AssignedItems.remove(it)

            if len(AssignedItems) > 0:
                for ii in range(0,len(AssignedItems)):
                    ret[agentIDs[-1]].append(AssignedItems[ii])
        else:
            pass

        #print('ret = ', ret)
        return  ret


    # assignments based on backtracking solution (self.AgentsAssignment1_2(...))
    # add location of each agent as start_pt
    # Criterion (assignment criterion):
    #       1 -- (default) assigned by agent capacity interms of agents' volumns and weights;
    #       2 -- assigned by equal number of items (for nightly scan)
    def AgentsAssignment1(self,start_pt,NIINs,agentIDs,Criterion=1,n=1,r=0,d=5):
        ret_AgentAssignment = self.AgentsAssignment1_2(start_pt,NIINs,agentIDs,Criterion,n,r,d)

        ret = {}
        for k in ret_AgentAssignment.keys():
            #print('**** k in 000 = ', k)
            items_k = ret_AgentAssignment[k]

            #print('items_k = ', items_k)
            #if k == 1 and len(items_k)>0: items_k.remove(start_pt)  # ret_AgentAssignment[1][0] is start_pt

            ##print('start_pt = ', start_pt)
            ##print('items_k = ', items_k)
            ret_OptimalSequence = self.OptimalSequence(str(start_pt),items_k,n,r,d)

            ##print('Here is 111 = ', ret_OptimalSequence[1])
            ret[k] = ret_OptimalSequence[1]
            ret[k][0] = (str(k),ret[k][0])

        return ret


    # Cluster analysis: multi-agents assignments
    # In application, item_locs only includes storage locations, NOT agent locations (no eqivalent_loc() is needed here)
    def Cluster_Analysis(self,item_locs=[],n_agents=1):
        # here, self.cur_DistMatrix must only include those items needed for Cluster Analysis
        item_locs_c = list(set(item_locs))
        #item_locs_c = item_locs
        self.update_cur_DistMatrix(leafnodeNames=item_locs_c)

        clustering2 = AgglomerativeClustering(n_clusters=n_agents, \
                                     linkage='complete', \
                                     affinity='precomputed', \
                                     connectivity=None, \
                                     compute_full_tree=True, \
                                     distance_threshold=None).fit(self.cur_DistMatrix) # "less than" distance_threshold

        idx2 = clustering2.labels_

        assignments = {}
        for agent in range(0,len(set(idx2))):
            #print(agent)
            assignments[agent] = []

        for i in item_locs_c:
            #print(i,item_locs_c.index(i))
            assignments[idx2[item_locs_c.index(i)]].append(i)

        return assignments

    # return a location in locs=[...] that is closest to cluster=[...] on average
    def closestLoc_cluster(self,locs=[],cluster=[]):
        if len(locs) == 0 or len(cluster) == 0: return None

        ret_loc = locs[0]
        eq_ret_loc = self.eqivalent_loc(ret_loc)
        ret_d = np.mean([self.cur_DistMatrix[eq_ret_loc][str(k)] for k in cluster])
        #print('ret_loc = ',ret_loc)
        #print('cluster = ',cluster)
        for a_loc in locs[1:]:
            eq_a_loc =  self.eqivalent_loc(a_loc)
            if ret_d > np.mean([self.cur_DistMatrix[eq_a_loc][k] for k in cluster]):
               ret_loc = a_loc
               ret_d = np.mean([self.cur_DistMatrix[eq_a_loc][k] for k in cluster])

        return ret_loc


    # (1) item assignments based on Cluster_Analysis (self.AgentsAssignment1_2(...))
    # (2) find optimal sequence for each item assignment in (1)
    # Here, agents = [('agent1','(91, 17)'),('agent2','(51, 17)'),...]
    #       r1: threshold for initial (without sided-locations) backtracking solution;
    #       r2: threshold for final (with sided-locations) backtracking solution
    def AgentsAssignment2(self,agents=[],item_locs=[],n_agents=1,n=1,r1=0,r2=0,d=5):
        if len(agents) == 0 or len(item_locs) == 0: return None

        # (items) assignments by Cluster Analysis
        # Note: In self.Cluster_Analysis(...), cur_DistMatrix is updated with only storage locations (item_locs)
        ret_AgentAssignment = self.Cluster_Analysis(item_locs,n_agents)

        ret = {}
        agentLocs = [x[1] for x in agents]
        for k in ret_AgentAssignment.keys():
            #print('**** k in 000 = ', k)
            items_k = ret_AgentAssignment[k]
            #print('items_k = ', items_k)

            # update self.cur_DistMatrix with adding agent locations
            ##print('agents = ', agents)
            #self.update_cur_DistMatrix(leafnodeNames = list(set([x[1] for x in agents] + item_locs)))
            self.update_cur_DistMatrix(leafnodeNames = list(set([self.eqivalent_loc(x[1]) for x in agents] + item_locs)))

            # select agent that is clostest to items_k
            aLoc = self.closestLoc_cluster(locs=agentLocs,cluster=items_k)
            agentLocs.remove(aLoc)

            ret_OptimalSequence = self.OptimalSequence2(str(aLoc),items_k,n,r1,r2,d)

            ##print('Here is 111 = ', str(aLoc), ret_OptimalSequence[1])
            ret[k] = ret_OptimalSequence[1]
            ret[k][0] = agents[[x[1] for x in agents].index(ret[k][0])]

        return ret

    # convert list of MELD locations to list of warehouse physical locations:
    # ["I213451AA","I213451BA",...] -> ['(51, 34)','(51, 34)',...]
    # Note: those duplicated locations are removed by list(set(list of warehouse locs))
    # here, HELD_Locs is, for example, ["I213451AA","I213451BA",...]
    def MELD_to_WH_LOCconverter(self,MELD_Locs=[]):
        warehouse_locs = []
        for loc in MELD_Locs:
            #print('***** loc = ',loc)
            aisle = loc[3:5]
            col = loc[5:7]
            #print('aisle = ',aisle, '   col = ',col)
            warehouse_loc = str((int(col), int(aisle)))
            warehouse_locs.append(warehouse_loc)

        return list(set(warehouse_locs))

    # input list of MELD locations:
    #       MELD_Locs=['I213451AA','I213753BA',...]
    # return a dictionary such as
    # {'(51, 34)':['I213451AA','I213451BA','I213451CA','I213451DA'],
    #  '(51, 35)':['I213551AA','I213551BA','I213551CA','I213551DA'], ...}
    # where '(51, 34)' is warehouse location while
    #       ['I213451AA','I213451BA','I213451CA','I213451DA'] are the MELD
    #       locations of the '(51, 34)'
    def WH_MELD_LocsDict(self,MELD_Locs=[]):
        warehouseMELD_locs = {}
        for loc in MELD_Locs:
            #print('***** loc = ',loc)
            aisle = loc[3:5]
            col = loc[5:7]
            #print('aisle = ',aisle, '   col = ',col)
            warehouse_loc = str((int(col), int(aisle)))

            if warehouse_loc in warehouseMELD_locs.keys():
                #warehouseMELD_locs[warehouse_loc].append(loc)
                warehouseMELD_locs[warehouse_loc] = warehouseMELD_locs[warehouse_loc] + [loc]

                if warehouse_loc in self.TwoSidedLeadNodes:
                    warehouseMELD_locs[str((int(col), int(aisle), "N"))] = warehouseMELD_locs[str((int(col), int(aisle), "N"))] + [loc+"N"]
                    warehouseMELD_locs[str((int(col), int(aisle), "S"))] = warehouseMELD_locs[str((int(col), int(aisle), "S"))] + [loc+"S"]
            else:
                warehouseMELD_locs[warehouse_loc] = [loc]

                if warehouse_loc in self.TwoSidedLeadNodes:
                    warehouseMELD_locs[str((int(col), int(aisle), "N"))] = [loc+"N"]
                    warehouseMELD_locs[str((int(col), int(aisle), "S"))] = [loc+"S"]

        return warehouseMELD_locs


    # Input: 1. WH_MELDlocs (dictionary) returned by WH_MELD_LocsDict(...)
    #        2. Optimal sequences for each Agent Assignment returned by AgentsAssignment2(...)
    # Return: a dictionary of MELD locations (of the order of input 2) for each agent assignment.
    def Optimal_MELDLocs_OLD(self,WH_MELDlocs={},OptimalAssignments={}):
        ret_MELDLocs = {}
        for k in OptimalAssignments.keys():
            agentID = 'Agent'+str(k+1)
            ret_MELDLocs[agentID] = [OptimalAssignments[k][0]]
            for WH_loc in OptimalAssignments[k][1:]:
                for MELD_loc in WH_MELDlocs[WH_loc]:
                    ret_MELDLocs[agentID].append(MELD_loc)

        return ret_MELDLocs


    # Input: 1. MELDlocs: list of MELD locations such as MELD_Locs=['I213451AA','I213753BA',...]
    #        2. Optimal sequences for each Agent Assignment returned by AgentsAssignment2(...)
    # Return: a dictionary of MELD locations (of the order of input 2) for each agent assignment.
    def Optimal_MELDLocs(self,MELDlocs=[],OptimalAssignments={}):
        WH_MELDlocs = self.WH_MELD_LocsDict(MELD_Locs=MELDlocs)

        ret_MELDLocs = {}
        for k in OptimalAssignments.keys():
            taskID = 'Task'+str(k+1)
            #ret_MELDLocs[taskID] = [OptimalAssignments[k][0]]
            ret_MELDLocs[taskID] = [(OptimalAssignments[k][0][0],self.Agent_PhyLoc_to_MELDLoc(OptimalAssignments[k][0][1]))]
            for WH_loc in OptimalAssignments[k][1:]:
                for MELD_loc in WH_MELDlocs[WH_loc]:
                    ret_MELDLocs[taskID].append(MELD_loc)

            # output should be in format {‘vehicle’:1, ‘locations’:[“I213453AA”, “I212962AA”, “I212491AA”]}
            # here, ret_MELDLocs looks like
            # {'Task1': [('agent1', '(52386.57073170732, 2107.018604651163)'), 'I214178AA', 'I214165AA', ...],
            #  'Task2': [('agent2', '(737.8390243902439, 702.339534883721)'), 'I210978AA', 'I210988AA', ...]}
            ret_MELDLocs2 = {}
            for k in ret_MELDLocs.keys():
                ret_MELDLocs2[k] = {}
                ret_MELDLocs2[k]['vehicle'] = ret_MELDLocs[k][0][0][5:]     # here, ret_MELDLocs[k][0][0] = 'agent1'
                ret_MELDLocs2[k]['locations'] = ret_MELDLocs[k][1:]

        #return ret_MELDLocs
        return ret_MELDLocs2

    # this is 'abstract' function which will be replaced with Agent_PhyLoc_to_MELDLoc(...) in child class
    def Agent_PhyLoc_to_MELDLoc(self,loc=''):
        return loc

    # this is 'abstract' function which will be replaced with Agent_MELDLoc_to_PhyLoc(...) in child class
    def Agent_MELDLoc_to_PhyLoc(self,loc=''):
        return loc

############################################
# class major_node
class major_Node:
    # Initialize the class
    def __init__(self, name:str, location:tuple, branch_Nodes_left=[],branch_Nodes_right=[],verticalSpace=0.0,horizontalSpace=0.0):
        self.name = name
        self.location = location
        self.branch_Nodes_left = branch_Nodes_left  # all left-side branch nodes
        self.branch_Nodes_right = branch_Nodes_right  # all right-side branch nodes
        self.verticalSpace = verticalSpace # vertical length between self.branch_Nodes_left and self.branch_Nodes_left
                                           # For example, self.verticalSpace = one-aisle vertical length between aisle 4 and aisle 6;
                                           # self.verticalSpace = 6-aisles vertical length between aisle 10 and aisle 17
        self.horizontalSpace = horizontalSpace # hrozontal length of space where major-node is located.
                                               # For example, self.horizontalSpace =  17 + 4/12 for n1, n4,n7 etc.
                                               # self.horizontalSpace =  22 + 7/12 for n3, n6, n9 etc.

    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name

    # major_Node reprewentation
    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.location))


# class branch_Node
class branch_Node:
    # Initialize the class
    def __init__(self, name:str, location:tuple, majorNode=[], cost=[],side=''):
        self.name = name
        self.location = location
        self.majorNodes = majorNode   # adjacent major nodes (crossing nodes)
        self.cost = cost    # Distances (costs) to major nodes: must be in the same oder as self.majorNotes
        self.side = side    # '':  this branch-node location has only one side to access
                            # 'N': Morthern side of this location
                            # 'S': Sorthern side of this location
                            # NOTE: in BAY2, only the locations in aisle 29 have two sides to access

    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name

    # branch_Node represetation
    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.location))


###########################################
# Warehouse Data Classes
class item:
    # Initialize the class
    def __init__(self, NIIN='', Description='', column=0, aisle=0):
        self.NIIN = NIIN
        self.Description = Description
        self.column = column
        self.aisle = aisle
        self.LENGTH = 0
        self.WIDTH = 0
        self.HEIGHT = 0
        self.WEIGHT = 0

    # item represetation
    def __repr__(self):
        return ('({0},{1},{2})'.format(self.NIIN, self.column, self.aisle))

class agent: # agent - AMR. AGV ot human user etc
    # Initialize the class
    def __init__(self,name='',capacity_volumn=0,capacity_weight=0):
        self.name = name
        self.capacity_volumn = capacity_volumn
        self.capacity_weight = capacity_weight

    def __repr__(self):
        return ('({0},{1},{2})'.format(self.name, self.capacity_volumn, self.capacity_weight))


class Warehouse_database:
    # Initialize the class
    def __init__(self,warehouseID=''):
        self.warehouseID = warehouseID
        self.items = {}
        self.agents = {}

    def __repr__(self):
        return ('({0})'.format(self.warehouseID))

    def add_item(self, i:item):
        self.items[i.NIIN] = i

    def add_agent(self, a:agent):
        self.agents[a.name] = a

    # load in D365 data (unfinished)
    def D365_data():
        return


############################################
# (Child) class BAY2

class BAY2(Warehouse):
    # create full self.gaph_df
    def create_graph_df(self):
        self.MajorNodesBAY2()  # create major nodes of self.graph_df based on BAY2
        self.MajorNodesDistancesBAY2()
        self.leaf_nodesBAY2()
        self.add_leafNodes_to_graph()
        self.make_undirected()  # this takes a little while
        self.TwoSidedLeadNodes_BAY2()

    # create major nodes of self.graph_df based on BAY4
    def MajorNodesBAY2(self):
        self.n_majorNodes = 36

        Vspace1 = 8/2            # vertical space between aisle 2 and aisle 3 or aisle 1 and aisle 2.
        Vspace2 = 6 + 6/12       # vertical space between aisle 4 and aisle 5 etc.
        Vspace3 = 17 + 6 + 4/12  # vertical space between aisle 10 and aisle 17
        Vspace4 = 6              # vertical space between aisle 18 and aisle 20 etc
        Vspace5 = 18 + 1/12      # vertical space between aisle 29 and aisle 34
        Vspace6 = (6 + 8/12)/2   # vertical space between aisle 41 and aisle 42 or aisle 42 and aisle 43.

        Hspace1 = 17 + 4/12     # horizontal space between (blokcs 1 and 3) column 84 and column 80 etc.
        Hspace2 = 18            # horizontal space between (blokcs 1 and 3) column 73 and column 68 etc.
        Hspace3 = 22 + 7/12     # horizontal space between (blokcs 1 and 3) column 61 and column 56 etc.
        Hspace4 = 20 + 5/12     # horizontal space between (blokcs 2) column 86 and column 79 etc.
        Hspace5 = 19 + 5/12     # horizontal space between (blokcs 2) column 61 and column 56 etc.

        majorNodes = []
        for i in range(1,self.n_majorNodes+1):
            MajorNodeName = 'n'+str(i)
            ##print(MajorNodeName)

            # calculate aisle number
            if i <= 9:
                aisle = int((i-1)/3)*3 + 2
            elif i <=14:
                aisle = 13.5
            elif i <= 22:
                aisle = int((i-14)/2)*3 + 16
            elif i <=27:
                aisle = 31.5
            else: # i >= 28
                aisle = int((i-28)/3)*3 + 36

            # calculate column number
            if i in [1,4,7,11,15,17,19,21,24,28,31,34]:
                col = 82
            elif i in [2,5,8,12,25,29,32,35]:
                col = 70.5
            elif i in [3,6,9,13,16,18,20,22,26,30,33,36]:
                col = 56.5
            elif i in [10,23]: # two gate nodes
                col = 92
            else: # [14,27] two gate nodes
                col = 50

            # calculate VSpace
            if i in [1,2,3]:
                VSpace = Vspace1
            elif i in [4,5,6,7,8,9,28,29,30,31,32,33]:
                VSpace = Vspace2
            elif i in [10,11,12,13,14]:
                VSpace = Vspace3
            elif i in [15,16,17,18,19,20,21,22]:
                VSpace = Vspace4
            elif i in [23,24,25,26,27]:
                VSpace = Vspace5
            elif i in [34,35,36]:
                VSpace = Vspace6

            # calulate HSpace
            if i in [1,4,7,11,24,28,31,34]:
                HSpace = Hspace1
            elif i in [2,5,8,12,25,29,32,35]:
                HSpace = Hspace2
            elif i in [3,6,9,13,26,30,33,36]:
                HSpace = Hspace3
            elif i in [15,17,19,21]:
                HSpace = Hspace4
            elif i in [16,18,20,22]:
                HSpace = Hspace5
            elif i in [10,14,23,27]:
                HSpace = 2 + 5/12

            #temp_majorode = major_Node(name=MajorNodeName,location=(col,aisle))
            temp_majorode = major_Node(name=MajorNodeName,location=(col,aisle),verticalSpace=VSpace,horizontalSpace=HSpace)
            majorNodes.append(temp_majorode)

        self.graph_df = pd.DataFrame(data=np.empty(shape=(len(majorNodes),len(majorNodes))),columns=[node.name for node in majorNodes])
        self.graph_df.index = self.graph_df.columns
        self.graph_df[0:][0:] = np.nan
        np.fill_diagonal(self.graph_df.values, 0)
        self.majorNodes = majorNodes

    # calcualte distances across major nodes in self.graph_df based on BAY2
    def MajorNodesDistancesBAY2(self):
        Bin1_Xlength = ((203+4/12)-(2+5/12)-(22+7/12)-18-(17+4/12)-(2+5/12))/30 # 203+4/12: total horizontal length; 30: 30 slots in each aisle. Bin1_Xlength=4.6861
        #Bin1_Ylength = 7.6389 # 2*Bin1_Ylength
        Bin1_Ylength = (42+12+42)/12 # 2*Bin2_Ylength: 8 (see BAY 2 - side view)
        Bin2_Ylength = 2*((68+8/12) - (6+4/12) - 6*4)/9 # 2*Bin2_Ylength: 8.518518518518519

        Hdistance_n1_n2 = 8*Bin1_Xlength + 18*0.5 + (17+4/12)*0.5
        Hdistance_n2_n3 = 8*Bin1_Xlength + (22+7/12)*0.5 + 18*0.5
        Hdistance_n13_n14 = 6*Bin1_Xlength + (22+7/12)*0.5 + (2+5/12)
        Hdistance_n15_n16 = Hdistance_n1_n2 + Hdistance_n2_n3

        Vdistance_n1_n4 = Bin1_Ylength + 8*0.5 + (6+6/12)*0.5
        Vdistance_n4_n7 = Bin1_Ylength + (6+6/12)*0.5 + (6+6/12)*0.5
        Vdistance_n7_n11 = Bin1_Ylength + (6+6/12)*0.5 + (17+6+4/12)*0.5
        Vdistance_n11_n15 = Bin2_Ylength + (17+6+4/12)*0.5 + (6+6/12)*0.5
        Vdistance_n15_n17 = Bin2_Ylength + 6*0.5 +  6*0.5
        Vdistance_n21_n24 = Bin2_Ylength*0.5 + 6*0.5 + (18+1/12)*0.5
        Vdistance_n24_n28 = Bin1_Ylength + (18+1/12)*0.5 + (6+6/12)*0.5

        # calculate between adjacent major nodes
        for i in range(1,self.n_majorNodes+1):
            #print('******', i, '*******')
            if i in [1,4,7,11,24,28,31,34]:
                self.connect('n'+str(i), 'n'+str(i+1), distance=Hdistance_n1_n2)
                if i in [11,24]:
                    ##print('i in 222 = ', i)
                    self.connect('n'+str(i-1), 'n'+str(i), distance=Hdistance_n1_n2 - 18*0.5)  # two 'gate nodes'

            if i in [2,5,8,12,25,29,32,35]:
                self.connect('n'+str(i), 'n'+str(i+1), distance=Hdistance_n2_n3)

            if i in [13,26]:
                self.connect('n'+str(i), 'n'+str(i+1), distance=Hdistance_n13_n14)  # two 'gate nodes'

            if i in [15,17,19,21]:
                self.connect('n'+str(i), 'n'+str(i+1), distance=Hdistance_n15_n16)

            if i in [1,2,3,31,32,33]:
                self.connect('n'+str(i), 'n'+str(i+3), distance=Vdistance_n1_n4)

            if i in [4,5,6,28,29,30]:
                self.connect('n'+str(i), 'n'+str(i+3), distance=Vdistance_n4_n7)

            if i in [7,8,9]:
                self.connect('n'+str(i), 'n'+str(i+4), distance=Vdistance_n7_n11)

            if i in [11]:
                self.connect('n'+str(i), 'n'+str(i+4), distance=Vdistance_n11_n15)

            if i in [13]:
                self.connect('n'+str(i), 'n'+str(i+3), distance=Vdistance_n11_n15)

            if i in [15,16,17,18,19,20]:
                self.connect('n'+str(i), 'n'+str(i+2), distance=Vdistance_n15_n17)

            if i in [21]:
                self.connect('n'+str(i), 'n'+str(i+3), distance=Vdistance_n21_n24)

            if i in [22]:
                self.connect('n'+str(i), 'n'+str(i+4), distance=Vdistance_n21_n24)

            if i in [24,25,26]:
                self.connect('n'+str(i), 'n'+str(i+4), distance=Vdistance_n24_n28)

        # make undirected
        self.make_undirected()

    # for each major node create all its leaf-nodes
    def leaf_nodesBAY2(self):
        Hlength_BAY4 = 203+4/12  # total horizontal length;

        # measures of block1/block3 structure
        Hlen_Block1_Blank1 = 17+4/12
        Hlen_Block1_Blank2 = 18
        Hlen_Block1_Blank3 = 22+7/12
        Hlen_Block1_Blanks = [Hlen_Block1_Blank1,Hlen_Block1_Blank2,Hlen_Block1_Blank3]

        N_BINs_Block1 = 30   # 30 bins (storage locations) in each aisle
        Hlen_Bin_Block1 = (Hlength_BAY4 - Hlen_Block1_Blank1 - Hlen_Block1_Blank2 - Hlen_Block1_Blank3 - (2+5/12)*2)/N_BINs_Block1   # 4.6861

        # measures of block2 structure
        Hlen_Block2_Blank1 = 20 + 5/12
        Hlen_Block2_Blank2 = 19 + 5/12
        Hlen_Block2_Blanks = [Hlen_Block2_Blank1,Hlen_Block2_Blank2]

        N_BINs_Block2 = 6 + 19 + 6
        Hlen_Bin_Block2 = (Hlength_BAY4 - (2+5/12)*2 - Hlen_Block2_Blank1 - Hlen_Block2_Blank2)/N_BINs_Block2  # 5.118279569892474

        # storage locations: rows and columns
        Block1_rows = [3,4,6,7,9,10]
        Block2_rows = [17,18,20,21,23,24,26,27,29]
        Block3_rows = [34,35,37,38,40,41]

        Block1or3_col1 = [91,90,89,88,87,86,85,84]
        Block1or3_col2 = [80,79,78,77,76,75,74,73]
        Block1or3_col3 = [68,67,66,65,64,63,62,61]
        Block1or3_col4 = [56,55,54,53,52,51]
        Block1or3_cols = [Block1or3_col1,Block1or3_col2,Block1or3_col3,Block1or3_col4]

        Block2_col1 = [i for i in range(91,85,-1)]  # [91,90,89,88,87,86]
        Block2_col2 = [i for i in range(79,60,-1)]  # [79,78,...,61]
        Block2_col3 = [i for i in range(56,50,-1)]  # [56,55,...,51]
        Block2_cols = [Block2_col1,Block2_col2,Block2_col3]

        Block2_col2_1 = [i for i in range(79,69,-1)] # [79,78,...,70]
        Block2_col2_2 = [i for i in range(69,60,-1)] # [69,78,...,61]
        Block2_col2_row11_16 = [Block2_col1,Block2_col2_1,Block2_col2_2,Block2_col3]

        c1 = 0
        c4 = 0
        c7 = 0
        c10 = 0
        c14 = 0
        c16 = 0
        c18 = 0
        c20 = 0
        c22 = 0
        c26 = 0
        c29 = 0
        c32 = 0
        for node in self.majorNodes:
            #print('****** ', node.name)
            if node.name in ['n1','n2','n3']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_cols[c1],rows=[3],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c1]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[c1+1],rows=[3],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c1]/2.0,left=False)
                c1 = c1 + 1
                continue

            if node.name in ['n4','n5','n6']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_cols[c4],rows=[4,6],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c4]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[c4+1],rows=[4,6],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c4]/2.0,left=False)
                c4 = c4 + 1
                continue

            if node.name in ['n7','n8','n9']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_cols[c7],rows=[7,9],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c7]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[c7+1],rows=[7,9],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c7]/2.0,left=False)
                c7 = c7 + 1
                continue

            if node.name in ['n10']:
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[0],rows=[10],majorNodeName=[node.name],\
                                                                BinLen=Hlen_Bin_Block1,HalfBlankLen=0.0,left=False)
                                                   # BinLen=Hlen_Bin_Block1,HalfBlankLen=0.0,left=False,HalfVlen_aisle=(17+6+4/12)/2.0)
                node.branch_Nodes_right = node.branch_Nodes_right + \
                    self.leafNodesCreater(cols=Block2_col2_row11_16[0],rows=[17],majorNodeName=[node.name],\
                                                  BinLen=Hlen_Bin_Block2,HalfBlankLen=0.0,left=False)
                                                  # BinLen=Hlen_Bin_Block2,HalfBlankLen=0.0,left=False,HalfVlen_aisle=(17+6+4/12)/2.0)

            if node.name in ['n11','n12','n13']: # here, c10 = 0 means 'n11'; c10 = 1 means 'n12' ; c10 = 2 means 'n13'
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_cols[c10],rows=[10],majorNodeName=[node.name],\
                                                               BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c10]/2.0,left=True)
                                                   #BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c10]/2.0,left=True,HalfVlen_aisle=(17+6+4/12)/2.0)
                node.branch_Nodes_left = node.branch_Nodes_left + \
                    self.leafNodesCreater(cols=Block2_col2_row11_16[c10],rows=[17],majorNodeName=[node.name],\
                                        BinLen=Hlen_Bin_Block2,HalfBlankLen=[Hlen_Block2_Blank1*0.7,0.0,Hlen_Block2_Blank2*0.7][c10],left=True)
                                    #BinLen=Hlen_Bin_Block2,HalfBlankLen=[Hlen_Block2_Blank1,0.0,Hlen_Block2_Blank2][c10]/2.0,left=True,HalfVlen_aisle=(17+6+4/12)/2.0)

                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[c10+1],rows=[10],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c10]/2.0,left=False)
                                                   #BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c10]/2.0,left=False,HalfVlen_aisle=(17+6+4/12)/2.0)
                node.branch_Nodes_right = node.branch_Nodes_right + \
                    self.leafNodesCreater(cols=Block2_col2_row11_16[c10+1],rows=[17],majorNodeName=[node.name],\
                                    BinLen=Hlen_Bin_Block2,HalfBlankLen=[Hlen_Block2_Blank1*0.3,0.0,Hlen_Block2_Blank2*0.3][c10],left=False)
                                    #BinLen=Hlen_Bin_Block2,HalfBlankLen=[Hlen_Block2_Blank1,0.0,Hlen_Block2_Blank2][c10]/2.0,left=False,HalfVlen_aisle=(17+6+4/12)/2.0)
                c10 = c10 + 1
                continue

            if node.name in ['n14']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_col4,rows=[10],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=0.0,left=True)
                                                   #BinLen=Hlen_Bin_Block1,HalfBlankLen=0.0,left=True,HalfVlen_aisle=(17+6+4/12)/2.0)
                node.branch_Nodes_left = node.branch_Nodes_left + \
                    self.leafNodesCreater(cols=Block2_col3,rows=[17],majorNodeName=[node.name],\
                                              BinLen=Hlen_Bin_Block2,HalfBlankLen=0.0,left=True)
                                                   #BinLen=Hlen_Bin_Block2,HalfBlankLen=0.0,left=True,HalfVlen_aisle=(17+6+4/12)/2.0)
                continue

            if node.name in ['n15','n16']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block2_cols[c14],rows=[18,20],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c14]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block2_cols[c14+1],rows=[18,20],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c14]/2.0,left=False)
                c14 = c14 + 1
                continue

            if node.name in ['n17','n18']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block2_cols[c16],rows=[21,23],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c16]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block2_cols[c16+1],rows=[21,23],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c16]/2.0,left=False)
                c16 = c16 + 1
                continue

            if node.name in ['n19','n20']:
                if node.name in ['n19']:
                    node.branch_Nodes_left = self.leafNodesCreater(cols=Block2_cols[c18],rows=[24],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c18]/2.0,left=True)
                else:
                    node.branch_Nodes_left = self.leafNodesCreater(cols=Block2_cols[c18],rows=[24,26],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c18]/2.0,left=True)

                node.branch_Nodes_right = self.leafNodesCreater(cols=Block2_cols[c18+1],rows=[24,26],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c18]/2.0,left=False)
                c18 = c18 + 1
                continue

            if node.name in ['n21','n22']:
                if node.name in ['n22']:
                    node.branch_Nodes_left = self.leafNodesCreater(cols=Block2_cols[c20],rows=[27,29],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c20]/2.0,left=True)
                    node.branch_Nodes_left =  node.branch_Nodes_left + \
                                                   self.leafNodesCreater2(cols=Block2_cols[c20],rows=[29],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c20]/2.0,left=True,side='N')

                node.branch_Nodes_right = self.leafNodesCreater(cols=Block2_cols[c20+1],rows=[27,29],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c20]/2.0,left=False)
                node.branch_Nodes_right = node.branch_Nodes_right + \
                                                   self.leafNodesCreater2(cols=Block2_cols[c20+1],rows=[29],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blanks[c20]/2.0,left=False,side='N')
                c20 = c20 + 1
                continue

            if node.name in ['n23']:
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_col1,rows=[34],majorNodeName=[node.name],\
                              BinLen=Hlen_Bin_Block1,HalfBlankLen=0.0,left=False,HalfVlen_aisle=(18+1/12)/2.0)
                continue

            if node.name in ['n24']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_col1,rows=[34],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blank1/2.0,left=True)

                node.branch_Nodes_right = self.leafNodesCreater(cols=Block2_col2_1,rows=[29],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blank1/2.0,left=False)
                node.branch_Nodes_right = node.branch_Nodes_right + \
                    self.leafNodesCreater(cols=Block1or3_col2,rows=[34],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blank1/2.0,left=False)
                node.branch_Nodes_right = node.branch_Nodes_right + \
                    self.leafNodesCreater2(cols=Block2_col2_1,rows=[29],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block2,HalfBlankLen=Hlen_Block2_Blank1/2.0,left=False,side='S')
                continue

            if node.name in ['n25','n26','n27']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block2_col2_row11_16[c22+1],rows=[29],majorNodeName=[node.name],BinLen=Hlen_Bin_Block2, \
                                                            HalfBlankLen=[0.0,Hlen_Block2_Blank2,0.0][c22]/2.0,left=True)
                node.branch_Nodes_left = node.branch_Nodes_left + \
                    self.leafNodesCreater(cols=Block1or3_cols[c22+1],rows=[34],majorNodeName=[node.name],BinLen=Hlen_Bin_Block1,\
                                          HalfBlankLen=[Hlen_Block1_Blank2,Hlen_Block1_Blank3,0.0][c22]/2.0,left=True)
                node.branch_Nodes_left = node.branch_Nodes_left + \
                                            self.leafNodesCreater2(cols=Block2_col2_row11_16[c22+1],rows=[29],majorNodeName=[node.name],BinLen=Hlen_Bin_Block2, \
                                                            HalfBlankLen=[0.0,Hlen_Block2_Blank2,0.0][c22]/2.0,left=True,side='S')

                if node.name in ['n25','n26']:
                    node.branch_Nodes_right = self.leafNodesCreater(cols=Block2_col2_row11_16[c22+2],rows=[29],majorNodeName=[node.name],\
                                            BinLen=Hlen_Bin_Block2,HalfBlankLen=[0.0,Hlen_Block2_Blank2,0.0][c22]/2.0,left=False)
                    node.branch_Nodes_right = node.branch_Nodes_right + \
                                          self.leafNodesCreater(cols=Block1or3_cols[c22+2],rows=[34],majorNodeName=[node.name], \
                                            BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c22+1]/2.0,left=False)
                    node.branch_Nodes_right = node.branch_Nodes_right + \
                                                self.leafNodesCreater2(cols=Block2_col2_row11_16[c22+2],rows=[29],majorNodeName=[node.name],\
                                                                      BinLen=Hlen_Bin_Block2,HalfBlankLen=[0.0,Hlen_Block2_Blank2,0.0][c22]/2.0,left=False,side='S')
                c22 = c22 + 1
                continue

            if node.name in ['n28','n29','n30']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_cols[c26],rows=[35,37],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c26]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[c26+1],rows=[35,37],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c26]/2.0,left=False)
                c26 = c26 + 1
                continue


            if node.name in ['n31','n32','n33']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_cols[c29],rows=[38,40],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c29]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[c29+1],rows=[38,40],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c29]/2.0,left=False)
                c29 = c29 + 1
                continue

            if node.name in ['n34','n35','n36']:
                node.branch_Nodes_left = self.leafNodesCreater(cols=Block1or3_cols[c32],rows=[41],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c32]/2.0,left=True)
                node.branch_Nodes_right = self.leafNodesCreater(cols=Block1or3_cols[c32+1],rows=[41],majorNodeName=[node.name],\
                                                   BinLen=Hlen_Bin_Block1,HalfBlankLen=Hlen_Block1_Blanks[c32]/2.0,left=False)
                c32 = c32 + 1
                continue


    # generate eqivalent location for those physical locations rather than storage locations (leaf-locations) and major-locations
    def eqivalent_loc(self,loc=''):
        #print('loc in eqivalent_loc = ',loc)
        assert loc in self.node_names() or literal_eval(loc)[0] > 50 and literal_eval(loc)[0] < 92
        assert loc in self.node_names() or literal_eval(loc)[1] > 0 and literal_eval(loc)[1] < 44

        Block1_rows = [2,5,8,11,12,13,14]
        Block2_rows = [15,16,19,22,25,28,30,31]
        Block3_rows = [32,33,36,39,42,43]

        Block1or3_col1 = [91,90,89,88,87,86,85,84]
        Block1or3_col2 = [80,79,78,77,76,75,74,73]
        Block1or3_col3 = [68,67,66,65,64,63,62,61]
        Block1or3_col4 = [56,55,54,53,52,51]
        Block1or3_cols = Block1or3_col1+Block1or3_col2+Block1or3_col3+Block1or3_col4

        Block2_col1 = [i for i in range(91,85,-1)]  # [91,90,89,88,87,86]
        Block2_col2 = [i for i in range(79,60,-1)]  # [79,78,...,61]
        Block2_col3 = [i for i in range(56,50,-1)]  # [56,55,...,51]
        Block2_cols = Block2_col1+Block2_col2+Block2_col3

        Space1_cols = [83,82,81]
        Space2_cols = [72,71,70,69]
        Space3_cols = [60,59,58,57]

        if loc in self.node_names():
            eq_loc = loc
        else:
            loc_tuple = literal_eval(loc)
            if int(loc_tuple[1]) in Block1_rows + Block3_rows: # for Blocks 1 and 3
                if int(loc_tuple[0]) in Block1or3_cols:
                    if  int(loc_tuple[1]) in [1]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(3)+')')
                    if  int(loc_tuple[1]) in [2,5,8]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(loc_tuple[1]+1)+')')
                    elif int(loc_tuple[1]) in [11,12]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(10)+')')
                    elif int(loc_tuple[1]) in [13,14]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(10)+')')
                        if int(loc_tuple[0]) in [52,51]:
                            eq_loc = 'n14'
                        elif int(loc_tuple[0]) in [91,90]:
                            eq_loc = 'n10'
                    elif int(loc_tuple[1]) in [32]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(34)+')')
                        if int(loc_tuple[0]) in [52,51]:
                            eq_loc = 'n27'
                        elif int(loc_tuple[0]) in [91,90]:
                            eq_loc = 'n23'
                    elif int(loc_tuple[1]) in [33]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(34)+')')
                    elif int(loc_tuple[1]) in [36,39,42]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(loc_tuple[1]-1)+')')
                    elif int(loc_tuple[1]) in [43]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(loc_tuple[1]-2)+')')
            elif int(loc_tuple[1]) in Block2_rows: # for Block 2
                if int(loc_tuple[0]) in Block2_cols:
                    if int(loc_tuple[1]) in [15,16]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(17)+')')
                    if int(loc_tuple[1]) in [19,22,25,28]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(loc_tuple[1]-1)+')')
                    if int(loc_tuple[1]) in [30,31]:
                        eq_loc = str('('+str(loc_tuple[0])+', '+str(34)+')')
                        if int(loc_tuple[0]) in [53,52,51]:
                            eq_loc = 'n27'
                        if int(loc_tuple[0]) in [68,69,70,71,72]:
                            eq_loc = 'n25'
                        if int(loc_tuple[0]) in [89,90,91]:
                            eq_loc = 'n23'

            # open-space locations
            if int(loc_tuple[0]) in list(set(Block2_col1)-set([91,90,89])):
                if int(loc_tuple[1]) in [26]:
                    eq_loc = 'n19'
                if int(loc_tuple[1]) in [27,28,29]:
                    eq_loc = 'n21'
                if int(loc_tuple[1]) in [30,31]:
                    eq_loc = 'n24'
            elif int(loc_tuple[0]) in [91,90,89]:
                if int(loc_tuple[1]) in [26,27]:
                    eq_loc = str('('+str(loc_tuple[0])+', '+str(24)+')')
                if int(loc_tuple[1]) in [28,29,30,31]:
                    eq_loc = 'n23'

            ##### space locations
            if int(loc_tuple[0]) in Space1_cols:
                if int(loc_tuple[1]) in [2,3]:
                    eq_loc = 'n1'
                elif int(loc_tuple[1]) in [4,5,6]:
                    eq_loc = 'n4'
                elif int(loc_tuple[1]) in [7,8,9,10]:
                    eq_loc = 'n7'
                elif int(loc_tuple[1]) in [11,12,13,14,15,16]:
                    eq_loc = 'n11'
                elif int(loc_tuple[1]) in [34,35,36,37]:
                    eq_loc = 'n28'
                elif int(loc_tuple[1]) in [38,39,40]:
                    eq_loc = 'n31'
                elif int(loc_tuple[1]) in [41,42,43]:
                    eq_loc = 'n34'
                if int(loc_tuple[1]) in [17,18,19,20]:
                    eq_loc = 'n15'
                elif int(loc_tuple[1]) in [21,22,23]:
                    eq_loc = 'n17'
                elif int(loc_tuple[1]) in [24,25,26]:
                    eq_loc = 'n19'
                elif int(loc_tuple[1]) in [27,28,29]:
                    eq_loc = 'n21'
                elif int(loc_tuple[1]) in [30,31,32,33]:
                    eq_loc = 'n24'
            elif int(loc_tuple[0]) in [80]:
                if int(loc_tuple[1]) in [17,18,19,20]:
                    eq_loc = 'n15'
                elif int(loc_tuple[1]) in [21,22,23]:
                    eq_loc = 'n17'
                elif int(loc_tuple[1]) in [24,25,26]:
                    eq_loc = 'n19'
                elif int(loc_tuple[1]) in [27,28,29]:
                    eq_loc = 'n21'
                elif int(loc_tuple[1]) in [30,31,32,33]:
                    eq_loc = 'n24'
            elif int(loc_tuple[0]) in Space2_cols:
                if int(loc_tuple[1]) in [2,3]:
                    eq_loc = 'n2'
                elif int(loc_tuple[1]) in [4,5,6]:
                    eq_loc = 'n5'
                elif int(loc_tuple[1]) in [7,8,9,10]:
                    eq_loc = 'n8'
                elif int(loc_tuple[1]) in [11,12,13,14,15,16]:
                    eq_loc = 'n12'
                elif int(loc_tuple[1]) in [30,31,32,33]:
                    eq_loc = 'n25'
                elif int(loc_tuple[1]) in [34,35,36,37]:
                    eq_loc = 'n29'
                elif int(loc_tuple[1]) in [38,39,40]:
                    eq_loc = 'n32'
                elif int(loc_tuple[1]) in [41,42,43]:
                    eq_loc = 'n35'
            elif int(loc_tuple[0]) in Space3_cols:
                if int(loc_tuple[1]) in [2,3]:
                    eq_loc = 'n3'
                elif int(loc_tuple[1]) in [4,5,6]:
                    eq_loc = 'n6'
                elif int(loc_tuple[1]) in [7,8,9,10]:
                    eq_loc = 'n9'
                elif int(loc_tuple[1]) in [11,12,13,14,15,16]:
                    eq_loc = 'n13'
                elif int(loc_tuple[1]) in [34,35,36,37]:
                    eq_loc = 'n30'
                elif int(loc_tuple[1]) in [38,39,40]:
                    eq_loc = 'n33'
                elif int(loc_tuple[1]) in [41,42,43]:
                    eq_loc = 'n36'
                if int(loc_tuple[1]) in [17,18,19,20]:
                    eq_loc = 'n16'
                elif int(loc_tuple[1]) in [21,22,23]:
                    eq_loc = 'n18'
                elif int(loc_tuple[1]) in [24,25,26]:
                    eq_loc = 'n20'
                elif int(loc_tuple[1]) in [27,28,29]:
                    eq_loc = 'n22'
                elif int(loc_tuple[1]) in [30,31,32,33]:
                    eq_loc = 'n26'

        return eq_loc

    # convert agent physical location to MELD agent location such as
    #               (90, 17) --> (20980,7264)
    # where (90, 17) = (column,aisle)
    #       (20980,7264) = (XCoordinate,YCoordinate) in millimeter
    # Note: 1 ft = 304.8mm
    def Agent_PhyLoc_to_MELDLoc(self,loc=''):
        assert literal_eval(loc)[0] > 50 and literal_eval(loc)[0] < 92
        assert literal_eval(loc)[1] > 0 and literal_eval(loc)[1] < 44

        try:
            loc_c = literal_eval(loc)
            XUnit = (203+4/12-2*(2+5/12))/(91-50)
            YUnit = (198+2/12)/43

            X = XUnit*(float(loc_c[0])-50-0.5)*304.8
            Y = YUnit*(43-float(loc_c[1])+0.5)*304.8

            return str((X,Y))
        except:
            print('An exception occurred: check if physical location', loc, 'is invalid.')


    # convert MELD agent location to agent physical location such as
    #               (20980,7264) --> (90, 17)
    # where (90, 17) = (column,aisle)
    #       (20980,7264) = (XCoordinate,YCoordinate) in millimeter
    # Note: 1 ft = 304.8mm
    def Agent_MELDLoc_to_PhyLoc(self,MELD_loc=''):
        try:
            MELD_loc_c = literal_eval(MELD_loc)
            XUnit = (203+4/12-2*(2+5/12))/(91-50)
            YUnit = (198+2/12)/43

            col = int(float(MELD_loc_c[0])/(304.8*XUnit)+50.5)
            aisle = int(43.5 - float(MELD_loc_c[1])/(304.8*YUnit))

            if col < 51: col = 51
            if col > 92: col = 92
            if aisle < 1: aisle = 1
            if aisle > 43: aisle = 43

            return str((col,aisle))
        except:
            print('An exception occurred: check if physical location', MELD_loc, 'is invalid.')

    # create self.TwoSidedLeadNodes.
    def TwoSidedLeadNodes_BAY2(self):
        Block2_col2 = [i for i in range(79,60,-1)]  # [79,78,...,61]
        Block2_col3 = [i for i in range(56,50,-1)]  # [56,55,...,51]

        for col in Block2_col2 + Block2_col3:
            self.TwoSidedLeadNodes.append(str((col,29)))
