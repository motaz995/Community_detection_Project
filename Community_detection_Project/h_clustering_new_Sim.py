from __future__ import division
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt 
import math 
from scipy.io import mmread
from networkx.algorithms.community.quality import modularity
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


def transfrom_list_of_sets_to_list_of_lists(list_comm):
    biglist=[list(e) for e in list_comm]
    return biglist

def get_communities_from_txt(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    comms=[line.split() for line in Lines]
    file1.close()
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
        
    
print("reading data....") 

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



print("data readed... !")

G = nx.parse_edgelist(Data,nodetype=int)
print("nodes : ",len(G.nodes()))
print("\nedges : ",len(G.edges()))
print("graph read")



def scoreN(n,Graph):
    Vn=set(list(Graph.neighbors(n)))
    s=0
    for no in Vn:
        Vno=set(list(Graph.neighbors(no)))
        s=s+(1/len(Vno))
    return s 

def score_sim(n,Graph):
    Vn=set(list(Graph.neighbors(n)))
    s=0
    for no in Vn:
        s=s+Graph[n][no]['weight']
def average_size(list_c):
    s=0
    for e in list_c:
        s=s+len(e)
    return s/len(list_c)
def average_density(Graph,list_of_comm):
    s=0
    for e in list_of_comm:
        s=s+nx.density(Graph.subgraph(e))
    return s/len(list_of_comm)

#calculate similarity
alpha=0.1
print("\nwith alpha = ",alpha,"\n")
print("calculating similarities...\n")
for (u,v) in G.edges():
    adjU=set(list(G.neighbors(u))).union({u})
    adjV=set(list(G.neighbors(v))).union({v})
    G[u][v]['weight']=((alpha*len(G.subgraph(adjU.intersection(adjV)).edges()))+((1-alpha)*len(adjU.intersection(adjV))))/((alpha*min(len(G.subgraph(adjU).edges()),len(G.subgraph(adjV).edges())))+((1-alpha)*min(len(adjU),len(adjV))))
print("similarities calculated ! \n")
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
def community_similarity(Graph,c1,c2):
    totalnei=set()
    for e in c1:
        totalnei=totalnei.union(set(list(Graph.neighbors(e))))
    return len(totalnei.intersection(c2))/min(len(totalnei),len(c2))
def copy_li(li):
    ll=[]
    for e in li:
        ll.append(e)
    return ll
#detection communities
print("detecting comms....")
current_clustering_list=init_commnuities(G)
current_modularity=modularity(G,current_clustering_list,weight='weight')
list_weights=Getlist_simlarity(G)
while True:
    if list_weights:
        maxi_weight=max(list_weights)
    else:
        break
    maxi_edges_sets=Get_Max_Edges_sets(G,maxi_weight)
    maxi_edges_sets_merged=list(nx1.connected_components(nx1.Graph(maxi_edges_sets)))
    previous_clustering_list=current_clustering_list
    previous_modularity=current_modularity
    current_clustering_list=Merge_edges(current_clustering_list+maxi_edges_sets_merged)
    current_modularity=modularity(G,current_clustering_list,weight='weight')
    if current_modularity<previous_modularity:
        break
    list_weights.remove(maxi_weight)

result=previous_clustering_list

print("transforming Communities....")
communities_nmi = [set(p) for p in communities]
print("comm transformed ")

res_omega_nmi=NodeClustering(result, graph=G, method_name="normalized_mutual_information")

communities_omega_nmi=NodeClustering(communities_nmi, graph=G, method_name="normalized_mutual_information")
print("\nthe number of communities detected by h_clust_new_sim algorithm",len(result))
print("\nthe number of real communities are : ",len(communities))

f1=evaluation.f1(communities_omega_nmi,res_omega_nmi)
print("\n\nF1 score = ",f1.score)
with open('C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/evaluation/eval_txt_code/f1_score.txt', 'w') as f:
    f.write(str(f1.score))

nmi=evaluation.normalized_mutual_information(communities_omega_nmi,res_omega_nmi)
print("\n\nNMI = ",nmi.score)
with open('C:/Users/w/Dropbox/thesis_motaz_ben_hassine/implementations/evaluation/eval_txt_code/nmi.txt', 'w') as f:
    f.write(str(nmi.score))
