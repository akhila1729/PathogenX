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
def test_model(data_loader, model, simLoss, epoch, mtlr = True):
    time_tensor = torch.tensor([])
    event_tensor = torch.tensor([])
    pred_tensor = torch.tensor([])
    risk_tensor = torch.tensor([])
    with torch.no_grad():
        running_loss = 0.0
        SIM_LOSS_LATENT = 0.0
        SIM_LOSS_GENOMIC = 0.0
        COX_LOSS = 0.0
        MTLR_LOSS = 0.0
        for x, time, event, clinical_data, time_val in data_loader:
            x, event, time = x.to(device), event.to(device), time.to(device)
            clinical_data = clinical_data.to(device)
            time_val = time_val.to(device)
            
            n_dead = event.sum()
            if n_dead == 0:
                continue
        
            out = model(data = x, clinical = clinical_data)
            
            # cox loss
            coxloss = CoxLoss(out['risk'], event, time_val)
            
            mtlr_loss = 0.0
            # mtlr loss 
            if mtlr:
                mtlr_loss = mtlr_neg_log_likelihood(out['logits'], time, average=True)
            
            #similarity_loss
            sim_loss_latent = simLoss(out['genomic_latent'], out['wsi_latent'])
            sim_loss_genomic = simLoss(out['genomic_latent'], out['genomics_from_pathomics'])
    
            loss = coxloss + mtlr_loss+sim_loss_genomic+sim_loss_latent
        
            running_loss = loss.item() + running_loss
            
            time_tensor = torch.cat((time_tensor, time_val.cpu()), dim=0)
            event_tensor = torch.cat((event_tensor, event.cpu()), dim=0)
            pred_tensor = torch.cat((pred_tensor, out['logits'].cpu()), dim=0)
            risk_tensor = torch.cat((risk_tensor, out['risk'].cpu()), dim=0)
            
            # cin = concordance_index(y_time.cpu(), -1*y_pred.cpu(), y_event.cpu())
            SIM_LOSS_LATENT = SIM_LOSS_LATENT + sim_loss_latent.item()
            SIM_LOSS_GENOMIC = SIM_LOSS_GENOMIC + sim_loss_genomic.item()
            COX_LOSS = COX_LOSS + coxloss.item()
            MTLR_LOSS = MTLR_LOSS + mtlr_loss.item()
            
        running_loss = running_loss/len(data_loader)
        
        writer.add_scalar("Loss/val", running_loss, epoch)
        writer.add_scalar("SIM_LOSS_LATENT/val", SIM_LOSS_LATENT, epoch)
        writer.add_scalar("SIM_LOSS_GENOMIC/val", SIM_LOSS_GENOMIC, epoch)
        writer.add_scalar("COX_LOSS/val", COX_LOSS, epoch)
        writer.add_scalar("MTLR_LOSS/val", MTLR_LOSS, epoch)
        
        y_time, y_event, y_pred =   time_tensor, event_tensor, pred_tensor 
        if mtlr:
            pred_risk = mtlr_risk(y_pred).cpu().numpy()
            time_sorted, indices_times = torch.sort(y_time)
            risk_tensor = risk_tensor[indices_times]
            event_ = y_event[indices_times]
              
            cin = concordance_index(time_sorted, -1*risk_tensor, event_)
            print(cin)
            writer.add_scalar("Cindex_Cox/val", cin, epoch)
            cin = concordance_index(time_tensor, -pred_risk, event_observed=event_tensor)
            writer.add_scalar("Cindex_MTLR/val", cin, epoch)
        else:
            cin = concordance_index(y_time, -1*y_pred, y_event)
        
        
        
    return running_loss, cin

def train_model(model, train_loader, val_loader,
                num_epochs = 1000, lr = 0.01, weight_decay = 0.,
                l1_reg = 0, batch_size = 10,
                device="cuda:0", l2_reg = 0.0, mtlr = True):
    # array conraining a list of losses
    
    loss_train = [] 
    loss_val = [] 
    old_cin = 0
    prev_loss = 10000
    optimizer = torch.optim.Adam(model.parameters(), lr)
    simLoss = nn.MSELoss()
    model = model.to(device)
    model.train()

    final_model = model 
    
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        SIM_LOSS_LATENT = 0.0
        SIM_LOSS_GENOMIC = 0.0
        COX_LOSS = 0.0
        MTLR_LOSS = 0.0
        for x, time, event, clinical_data, time_val in train_loader:
            
            x, event, time = x.to(device), event.to(device), time.to(device)
            clinical_data = clinical_data.to(device)
            time_val = time_val.to(device)
            
            n_dead = event.sum()
            if n_dead == 0:
                continue
            if (torch.isnan(x).any() or torch.isnan(event).any() or torch.isnan(time).any()):
                continue
            
            out = model(data = x, clinical = clinical_data)
        
            # cox loss
            coxloss = CoxLoss(out['risk'], event, time_val)
            
            mtlr_loss = 0.0
            # mtlr loss 
            if mtlr:
                mtlr_loss = mtlr_neg_log_likelihood(out['logits'], time, average=True)
            
            #similarity_loss
            
            sim_loss_latent = simLoss(out['genomic_latent'], out['wsi_latent'])
            
            sim_loss_genomic = simLoss(out['genomic_latent'], out['genomics_from_pathomics'])
    
            loss = coxloss + mtlr_loss+sim_loss_genomic+sim_loss_latent
            
#             # Add L1 regularization to the the first layer and compute loss
#             l1_loss = torch.tensor(0.).cuda()
#             for param in model.parameters():
#                 l1_loss += torch.norm(param, 1)
#                 break
#             loss = loss + l1_loss * l1_reg

            SIM_LOSS_LATENT = SIM_LOSS_LATENT + sim_loss_latent.item()
            SIM_LOSS_GENOMIC = SIM_LOSS_GENOMIC + sim_loss_genomic.item()
            COX_LOSS = COX_LOSS + coxloss.item()
            MTLR_LOSS = MTLR_LOSS + mtlr_loss.item()
            
            running_loss = running_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        running_loss = running_loss/len(train_loader)  
        SIM_LOSS_LATENT = SIM_LOSS_LATENT/len(train_loader)
        SIM_LOSS_GENOMIC = SIM_LOSS_GENOMIC/len(train_loader)
        COX_LOSS = COX_LOSS/len(train_loader)
        MTLR_LOSS = MTLR_LOSS/len(train_loader)
        
        loss_train.append(running_loss)
        print(running_loss)
        
        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("SIM_LOSS_LATENT/train", SIM_LOSS_LATENT, epoch)
        writer.add_scalar("SIM_LOSS_GENOMIC/train", SIM_LOSS_GENOMIC, epoch)
        writer.add_scalar("COX_LOSS/train", COX_LOSS, epoch)
        writer.add_scalar("MTLR_LOSS/train", MTLR_LOSS, epoch)
        writer.flush()
        # tr, c = test_model(train_loader, model)
        # print("Train Data", c)
        val_loss, cin = test_model(val_loader, model, simLoss, epoch)
        loss_val.append(val_loss)
        if(prev_loss>val_loss):
            final_model = model
            torch.save(model.state_dict(), 'exp1.p')
            prev_loss = val_loss
            
        print("Val Data", cin)
    final_model.eval()
    return final_model, loss_train, loss_val

