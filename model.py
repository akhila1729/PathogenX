import numpy as np
import matplotlib.pyplot as plt
import pickle
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
import pandas as pd
import numpy as np

import glob 
import math

import torch # For building the networks 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam
import torch.optim as optim
import torch.nn.functional as F
from nystrom_attention import NystromAttention

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score
from lifelines.utils import concordance_index
# import torchvision.transforms as transforms
from PIL import Image

# from TCGA_preprocess import TCGA_csv_process\
import sys

#Feature extraction
# from utils import perform_clustering, select_representative_images, extract_features
# import torchvision.models as models
import openslide as op
import cv2
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8, #8
            heads = 2, #8
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        a = self.attn(self.norm(x))
        x= x + a

        return x, None


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x



class TransMIL(nn.Module):
    def __init__(self):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
     

    def forward(self, data, encoder =0):

        h = data #[B, n, 1024]
        
        if encoder==1:
            h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
    
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
   
        #---->Translayer x1
        h, attn_scores1= self.layer1(h) #[B, N, 512]
        
        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h, attn_scores2 = self.layer2(h) #[B, N, 512]
        
        h = self.norm(h)
        return h[:,0], h[:,1:] #cls_token, sequence
        

class TransSurv(nn.Module):
    def __init__(self, n_classes):
        super(TransSurv, self).__init__()
        self.transmil_encoder = TransMIL()
        self.transmil_decoder = TransMIL()
        self.n_classes = n_classes
        self._fc2 = nn.Linear(512, 1)
#         self._fc4 = nn.Linear(128,1)
        self._fc_latent = nn.Linear(760, 512)
        self.mtlr = MTLR(in_features=512, num_time_bins=self.n_classes)
        self._fc3 = nn.Linear(256, self.n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        
        h, path_encoded = self.transmil_encoder(h, encoder=1)
        
        wsi_latent = h
        
        #genomic data
        genomic = kwargs['clinical'].float()
        genomic_latent = self._fc_latent(genomic) #(512, 1)
        
        genomic_from_pathomics,_ = self.transmil_decoder(path_encoded)
            

        risk = self._fc2(genomic_latent) # (512 -> 1) cox risk
#         risk = self._fc4(risk)
    
        logits = self.mtlr(h) #(512 -> 1) mtlr risk
        
        
        out = {"genomic_latent" : genomic_latent,
              "wsi_latent" : wsi_latent,
               "genomics_from_pathomics" : genomic_from_pathomics,
               "risk" : risk,
               "logits" : logits
              }
        return out
    
"""
Loss :
|genomics_from_pathomics-genomic_latent| + |genomic_latent-wsi_latent| + |cox_loss| + |mtlr_loss|
"""