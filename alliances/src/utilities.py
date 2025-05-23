import pandas as pd, numpy as np, datetime, random, os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support
#import requests


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def create_onehot_vector(size, index):
    vector = np.zeros(size)
    vector[index] = 1
    return vector

