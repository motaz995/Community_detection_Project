from __future__ import division
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt 
import math 
from scipy.io import mmread
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community.quality import performance
import random
from distinctipy import distinctipy
from networkx.algorithms import community
from cdlib import evaluation,algorithms,NodeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from cdlib import readwrite
from cdlib import readwrite
import numpy as np
import csv
import networkx as nx1
import networkx 
from networkx.algorithms.components.connected import connected_components
from igraph import clustering
#from sklearn.preprocessing import MultiLabelBinarizer  
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder



def transfrom_list_of_sets_to_list_of_lists(list_comm):
    biglist=[]
    for e in list_comm:
        sublist=list(e)
        biglist.append(sublist)
    return biglist

def get_communities_from_txt(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    comms=[]
    for line in Lines:
        comms.append(line.split())
    return comms

def tansform_comm_labsl_str(comm):
    l1=[]
    for e in comm:
        subl1=[]
        for el in e:
            subl1.append(str(el))
        l1.append(subl1)
    return l1
        
def tansform_comm_labsl_int(comm):
    l1=[]
    for e in comm:
        subl1=[]
        for el in e:
            subl1.append(int(el))
        l1.append(subl1)
    return l1            
        
    

print("getting data...")
#real data set 
#Data=open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/amazon/Amazon.csv","r")
#communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/amazon/amazon_Truth.json").communities

Data = open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/karate/karate.csv","r")
communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/karate/karate_Truth.json").communities

#Data = open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/emailEuCore/emailEucore.csv","r")
#communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/emailEuCore/email_Truth.json").communities

#Data = open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/football/football.csv","r")
#communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/football/football_truth.json").communities

#Data = open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/dolphins/dolphins.csv","r")
#communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/dolphins/dolphins_truth.json").communities

#Data = open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/Books/Books.csv","r")
#communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/Books/Books_truth.json").communities

#Data = open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/cora/cora.csv","r")
#communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/cora/cora_truth.json").communities


#Data = open("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/citeseer/citeseer.csv","r")
#communities=readwrite.read_community_json("C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/datasets/citeseer/citeseer_truth.json").communities


print("data got, creating graph now....")
G = nx.parse_edgelist(Data,nodetype=int)
print("nodes : ",len(G.nodes()))
print("\nedges : ",len(G.edges()))
print("graph  created")

#calculate similarity
print("calculating similarities")
for (u,v) in G.edges():
    adjU=set(list(G.neighbors(u))).union({u})
    adjV=set(list(G.neighbors(v))).union({v})
    G[u][v]['weight']=len(adjU.intersection(adjV))/math.sqrt(len(adjU)*len(adjV))
    
    
print("similarity done...")
def average_density(Graph,list_of_comm):
    s=0
    for e in list_of_comm:
        s=s+nx.density(Graph.subgraph(e))
    return s/len(list_of_comm)  

def Getlist_simlarity(Graph):
    sim_list=[]
    for (u,v,d) in Graph.edges(data=True):
        sim_list.append(d['weight'])
    return list(dict.fromkeys(sim_list))

def Get_Max_Edges_sets(Graph,maxi_weight):
    maxi_edges=[]
    for (u,v,d) in Graph.edges(data=True):
        if d['weight'] == maxi_weight:
            maxi_edges.append((u,v))
    return maxi_edges

def exist_intersection(list_set):
    for e1 in list_set:
        for e2 in list_set:
            if e1!=e2 and len(e1.intersection(e2))!=0:
                return True
    return False

def list_set_distinct(list_set):
    result=[]
    for e in list_set:
        if e not in result:
            result.append(e)
    return result

def to_edges(l):
    it = iter(l)
    last = next(it)
    for current in it:
        yield last, current
        last = current 
def to_graph(l):
    gr = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        gr.add_nodes_from(part)
        # it also imlies a number of edges:
        gr.add_edges_from(to_edges(part))
    return gr

def Merge_edges(List_edges):
    gra = to_graph(List_edges)
    return list(connected_components(gra))
    

def init_commnuities(Graph):
    Com=[]
    for node in Graph.nodes():
        Com.append({node})
    return Com
def copy_li(li):
    ll=[]
    for e in li:
        ll.append(e)
    return ll
def get_max_attached_community(nm, comms):
    el1=list(nm)[0]
    el2=list(nm)[1]
    att=[e for e in comms if el1 in e or el2 in e]
    att=sorted(att, key=lambda comm: len(G.subgraph(comm).edges()))
    if att:
        return att[0]

#detect communities  
print("detecting comms....")
current_clustering_list=init_commnuities(G)
current_modularity=modularity(G,current_clustering_list,weight='weight')
list_weights=Getlist_simlarity(G)
#print("itterations : \n")
while True:
    #print("'crurrent clustering' : ",current_clustering_list," modularity = ",current_modularity,"\n")
    if list_weights:
        maxi_weight=max(list_weights)
    else:
        break
    maxi_edges_sets=Get_Max_Edges_sets(G,maxi_weight)
    maxi_edges_sets_merged=list(nx1.connected_components(nx1.Graph(maxi_edges_sets)))
    #print("edges with smililariy",maxi_weight,"to merge with before : ",maxi_edges_sets," the result of merge is after \n")
    previous_clustering_list=current_clustering_list
    previous_modularity=current_modularity
    current_clustering_list=Merge_edges(current_clustering_list+maxi_edges_sets_merged)
    current_modularity=modularity(G,current_clustering_list,weight='weight')
    if current_modularity<previous_modularity:
        #print("'crurrent clustering' : ",current_clustering_list," modularity = ",current_modularity,"\n")
        break
    list_weights.remove(maxi_weight)
if current_modularity>=previous_modularity:
    result=current_clustering_list
    result_mod=current_modularity
else:
    result=previous_clustering_list
    result_mod=previous_modularity

print("communtites  founded ! ")


print("calculating NMI....")


communities = [set(p) for p in communities]

res_omega_nmi=NodeClustering(result, graph=G, method_name="normalized_mutual_information")

communities_omega_nmi=NodeClustering(communities, graph=G, method_name="normalized_mutual_information")


print("\nthe number of communities detected by h_clust algorithm",len(result))


print("\nthe number of real communities are : ",len(communities))


print("f1-score....")
#f1 score
f1=evaluation.f1(communities_omega_nmi,res_omega_nmi)
print("\n\nF1 score = ",f1.score)


#Nmi

nmi=evaluation.normalized_mutual_information(communities_omega_nmi,res_omega_nmi)
print("\n\nNMI = ",nmi.score)
