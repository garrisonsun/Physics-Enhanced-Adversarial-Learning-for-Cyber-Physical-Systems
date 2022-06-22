# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:57:56 2021

@author: Guoxin
"""
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter
save_path = 'EvaluationResults'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

###############################################################################
# =============================================================================
def dataExtractor(file_id, detector_type):
    df = pd.read_csv(file_id)
    mFDI = []
    vFDIAcce = []
    vFDISpeed = []
    vFDILoc = []
    vFDIAcceSpeed = []
    vFDIAcceLoc = []
    vFDISpeedLoc = []
    vFDIAll = []
    
    for idx, item in enumerate(df['chan_id']):
        # print(idx, item)
        if 'progAttack_rule_Acce5' in item:
            # print(df[df['chan_id']==item].index)
            mFDI.append(df.iloc[[idx]])
        if 'progAttack_Acce_' in item:
            # print(df[df['chan_id']==item].index)
            vFDIAcce.append(df.iloc[[idx]])
        if 'progAttack_Speed_' in item:
            # print(df[df['chan_id']==item].index)
            vFDISpeed.append(df.iloc[[idx]])
        if 'progAttack_Loc_' in item:
            # print(df[df['chan_id']==item].index)
            vFDILoc.append(df.iloc[[idx]])
        if 'progAttack_AcceLoc_' in item:
            # print(df[df['chan_id']==item].index)
            vFDIAcceLoc.append(df.iloc[[idx]])
        if 'progAttack_AcceSpeed_' in item:
            # print(df[df['chan_id']==item].index)
            vFDIAcceSpeed.append(df.iloc[[idx]])
        if 'progAttack_SpeedLoc_' in item:
            # print(df[df['chan_id']==item].index)
            vFDISpeedLoc.append(df.iloc[[idx]])
        if 'progAttack_AcceSpeedLoc_' in item:
            # print(df[df['chan_id']==item].index)
            vFDIAll.append(df.iloc[[idx]])
        
    ################################################################################    
    mFDI_predictorONLY_precsionn=[]
    mFDI_predictorONLY_recall=[]
    mFDI_predictorONLY_f1=[]
    
    mFDI_PCC_precsionn=[]
    mFDI_PCC_recall=[]
    mFDI_PCC_f1=[]
    
    mFDI_DAD_precsionn=[]
    mFDI_DAD_recall=[]
    mFDI_DAD_f1=[]
    
    mFDI_predictorONLY_precsionn_adv=[]
    mFDI_predictorONLY_recall_adv=[]
    mFDI_predictorONLY_f1_adv=[]
    
    mFDI_PCC_precsionn_adv=[]
    mFDI_PCC_recall_adv=[]
    mFDI_PCC_f1_adv=[]
    
    mFDI_DAD_precsionn_adv=[]
    mFDI_DAD_recall_adv=[]
    mFDI_DAD_f1_adv=[]
    for item in mFDI:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            mFDI_predictorONLY_precsionn.append(tmp['precision'])
            mFDI_predictorONLY_recall.append(tmp['recall'])
            mFDI_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            mFDI_PCC_precsionn.append(tmp['precision'])
            mFDI_PCC_recall.append(tmp['recall'])
            mFDI_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            mFDI_DAD_precsionn.append(tmp['precision'])
            mFDI_DAD_recall.append(tmp['recall'])
            mFDI_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            mFDI_predictorONLY_precsionn_adv.append(tmp['precision'])
            mFDI_predictorONLY_recall_adv.append(tmp['recall'])
            mFDI_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            mFDI_PCC_precsionn_adv.append(tmp['precision'])
            mFDI_PCC_recall_adv.append(tmp['recall'])
            mFDI_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            mFDI_DAD_precsionn_adv.append(tmp['precision'])
            mFDI_DAD_recall_adv.append(tmp['recall'])
            mFDI_DAD_f1_adv.append(tmp['f1-score'])        
    mFDI_predictorONLY_precsion_mean=np.mean(mFDI_predictorONLY_precsionn)
    mFDI_predictorONLY_recall_mean=np.mean(mFDI_predictorONLY_recall)
    print('mFDI_predictorONLY_recall_mean: ',mFDI_predictorONLY_recall_mean)
    mFDI_predictorONLY_f1_mean=np.mean(mFDI_predictorONLY_f1)
    print('mFDI_predictorONLY_f1_mean: ',mFDI_predictorONLY_f1_mean)
    
    mFDI_PCC_precsionn_mean=np.mean(mFDI_PCC_precsionn)
    mFDI_PCC_recall_mean=np.mean(mFDI_PCC_recall)
    print('mFDI_PCC_recall_mean: ',mFDI_PCC_recall_mean)
    mFDI_PCC_f1_mean=np.mean(mFDI_PCC_f1)
    print('mFDI_PCC_f1_mean: ',mFDI_PCC_f1_mean)
    
    mFDI_DAD_precsionn_mean=np.mean(mFDI_DAD_precsionn)
    mFDI_DAD_recall_mean=np.mean(mFDI_DAD_recall)
    print('mFDI_DAD_recall_mean: ',mFDI_DAD_recall_mean)
    mFDI_DAD_f1_mean=np.mean(mFDI_DAD_f1)
    print('mFDI_DAD_f1_mean: ',mFDI_DAD_f1_mean)
    
    mFDI_predictorONLY_precsionn_adv_mean=np.mean(mFDI_predictorONLY_precsionn_adv)
    mFDI_predictorONLY_recall_adv_mean=np.mean(mFDI_predictorONLY_recall_adv)
    print('mFDI_predictorONLY_recall_adv_mean: ',mFDI_predictorONLY_recall_adv_mean)
    mFDI_predictorONLY_f1_adv_mean=np.mean(mFDI_predictorONLY_f1_adv)
    print('mFDI_predictorONLY_f1_adv_mean: ',mFDI_predictorONLY_f1_adv_mean)
    
    mFDI_PCC_precsionn_adv_mean=np.mean(mFDI_PCC_precsionn_adv)
    mFDI_PCC_recall_adv_mean=np.mean(mFDI_PCC_recall_adv)
    print('mFDI_PCC_recall_adv_mean: ',mFDI_PCC_recall_adv_mean)
    mFDI_PCC_f1_adv_mean=np.mean(mFDI_PCC_f1_adv)
    print('mFDI_PCC_f1_adv_mean: ',mFDI_PCC_f1_adv_mean)
    
    mFDI_DAD_precsionn_adv_mean=np.mean(mFDI_DAD_precsionn_adv)
    mFDI_DAD_recall_adv_mean=np.mean(mFDI_DAD_recall_adv)
    print('mFDI_DAD_recall_adv_mean: ',mFDI_DAD_recall_adv_mean)
    mFDI_DAD_f1_adv_mean=np.mean(mFDI_DAD_f1_adv)
    print('mFDI_DAD_f1_adv_mean: ',mFDI_DAD_f1_adv_mean)
    
    mFDI_predictorONLY_precsion_std=np.std(mFDI_predictorONLY_precsionn)
    mFDI_predictorONLY_recall_std=np.std(mFDI_predictorONLY_recall)
    mFDI_predictorONLY_f1_std=np.std(mFDI_predictorONLY_f1)
    
    mFDI_PCC_precsionn_std=np.std(mFDI_PCC_precsionn)
    mFDI_PCC_recall_std=np.std(mFDI_PCC_recall)
    mFDI_PCC_f1_std=np.std(mFDI_PCC_f1)
    
    mFDI_DAD_precsionn_std=np.std(mFDI_DAD_precsionn)
    mFDI_DAD_recall_std=np.std(mFDI_DAD_recall)
    mFDI_DAD_f1_std=np.std(mFDI_DAD_f1)
    
    mFDI_predictorONLY_precsionn_adv_std=np.std(mFDI_predictorONLY_precsionn_adv)
    mFDI_predictorONLY_recall_adv_std=np.std(mFDI_predictorONLY_recall_adv)
    mFDI_predictorONLY_f1_adv_std=np.std(mFDI_predictorONLY_f1_adv)
    
    mFDI_PCC_precsionn_adv_std=np.std(mFDI_PCC_precsionn_adv)
    mFDI_PCC_recall_adv_std=np.std(mFDI_PCC_recall_adv)
    mFDI_PCC_f1_adv_std=np.std(mFDI_PCC_f1_adv)
    
    mFDI_DAD_precsionn_adv_std=np.std(mFDI_DAD_precsionn_adv)
    mFDI_DAD_recall_adv_std=np.std(mFDI_DAD_recall_adv)
    mFDI_DAD_f1_adv_std=np.std(mFDI_DAD_f1_adv)
    ###############################################################################        
    
    ################################################################################    
    vFDIAcce_predictorONLY_precsionn=[]
    vFDIAcce_predictorONLY_recall=[]
    vFDIAcce_predictorONLY_f1=[]
    
    vFDIAcce_PCC_precsionn=[]
    vFDIAcce_PCC_recall=[]
    vFDIAcce_PCC_f1=[]
    
    vFDIAcce_DAD_precsionn=[]
    vFDIAcce_DAD_recall=[]
    vFDIAcce_DAD_f1=[]
    
    vFDIAcce_predictorONLY_precsionn_adv=[]
    vFDIAcce_predictorONLY_recall_adv=[]
    vFDIAcce_predictorONLY_f1_adv=[]
    
    vFDIAcce_PCC_precsionn_adv=[]
    vFDIAcce_PCC_recall_adv=[]
    vFDIAcce_PCC_f1_adv=[]
    
    vFDIAcce_DAD_precsionn_adv=[]
    vFDIAcce_DAD_recall_adv=[]
    vFDIAcce_DAD_f1_adv=[]
    for item in vFDIAcce:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            # 
            vFDIAcce_predictorONLY_precsionn.append(tmp['precision'])
            vFDIAcce_predictorONLY_recall.append(tmp['recall'])
            vFDIAcce_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcce_PCC_precsionn.append(tmp['precision'])
            vFDIAcce_PCC_recall.append(tmp['recall'])
            vFDIAcce_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcce_DAD_precsionn.append(tmp['precision'])
            vFDIAcce_DAD_recall.append(tmp['recall'])
            vFDIAcce_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcce_predictorONLY_precsionn_adv.append(tmp['precision'])
            vFDIAcce_predictorONLY_recall_adv.append(tmp['recall'])
            vFDIAcce_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcce_PCC_precsionn_adv.append(tmp['precision'])
            vFDIAcce_PCC_recall_adv.append(tmp['recall'])
            vFDIAcce_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcce_DAD_precsionn_adv.append(tmp['precision'])
            vFDIAcce_DAD_recall_adv.append(tmp['recall'])
            vFDIAcce_DAD_f1_adv.append(tmp['f1-score'])        
    vFDIAcce_predictorONLY_precsion_mean=np.mean(vFDIAcce_predictorONLY_precsionn)
    vFDIAcce_predictorONLY_recall_mean=np.mean(vFDIAcce_predictorONLY_recall)
    vFDIAcce_predictorONLY_f1_mean=np.mean(vFDIAcce_predictorONLY_f1)
    
    vFDIAcce_PCC_precsionn_mean=np.mean(vFDIAcce_PCC_precsionn)
    vFDIAcce_PCC_recall_mean=np.mean(vFDIAcce_PCC_recall)
    vFDIAcce_PCC_f1_mean=np.mean(vFDIAcce_PCC_f1)
    
    vFDIAcce_DAD_precsionn_mean=np.mean(vFDIAcce_DAD_precsionn)
    vFDIAcce_DAD_recall_mean=np.mean(vFDIAcce_DAD_recall)
    vFDIAcce_DAD_f1_mean=np.mean(vFDIAcce_DAD_f1)
    
    vFDIAcce_predictorONLY_precsionn_adv_mean=np.mean(vFDIAcce_predictorONLY_precsionn_adv)
    vFDIAcce_predictorONLY_recall_adv_mean=np.mean(vFDIAcce_predictorONLY_recall_adv)
    vFDIAcce_predictorONLY_f1_adv_mean=np.mean(vFDIAcce_predictorONLY_f1_adv)
    
    vFDIAcce_PCC_precsionn_adv_mean=np.mean(vFDIAcce_PCC_precsionn_adv)
    vFDIAcce_PCC_recall_adv_mean=np.mean(vFDIAcce_PCC_recall_adv)
    vFDIAcce_PCC_f1_adv_mean=np.mean(vFDIAcce_PCC_f1_adv)
    
    vFDIAcce_DAD_precsionn_adv_mean=np.mean(vFDIAcce_DAD_precsionn_adv)
    vFDIAcce_DAD_recall_adv_mean=np.mean(vFDIAcce_DAD_recall_adv)
    vFDIAcce_DAD_f1_adv_mean=np.mean(vFDIAcce_DAD_f1_adv)
    
    vFDIAcce_predictorONLY_precsion_std=np.std(vFDIAcce_predictorONLY_precsionn)
    vFDIAcce_predictorONLY_recall_std=np.std(vFDIAcce_predictorONLY_recall)
    vFDIAcce_predictorONLY_f1_std=np.std(vFDIAcce_predictorONLY_f1)
    
    vFDIAcce_PCC_precsionn_std=np.std(vFDIAcce_PCC_precsionn)
    vFDIAcce_PCC_recall_std=np.std(vFDIAcce_PCC_recall)
    vFDIAcce_PCC_f1_std=np.std(vFDIAcce_PCC_f1)
    
    vFDIAcce_DAD_precsionn_std=np.std(vFDIAcce_DAD_precsionn)
    vFDIAcce_DAD_recall_std=np.std(vFDIAcce_DAD_recall)
    vFDIAcce_DAD_f1_std=np.std(vFDIAcce_DAD_f1)
    
    vFDIAcce_predictorONLY_precsionn_adv_std=np.std(vFDIAcce_predictorONLY_precsionn_adv)
    vFDIAcce_predictorONLY_recall_adv_std=np.std(vFDIAcce_predictorONLY_recall_adv)
    vFDIAcce_predictorONLY_f1_adv_std=np.std(vFDIAcce_predictorONLY_f1_adv)
    
    vFDIAcce_PCC_precsionn_adv_std=np.std(vFDIAcce_PCC_precsionn_adv)
    vFDIAcce_PCC_recall_adv_std=np.std(vFDIAcce_PCC_recall_adv)
    vFDIAcce_PCC_f1_adv_std=np.std(vFDIAcce_PCC_f1_adv)
    
    vFDIAcce_DAD_precsionn_adv_std=np.std(vFDIAcce_DAD_precsionn_adv)
    vFDIAcce_DAD_recall_adv_std=np.std(vFDIAcce_DAD_recall_adv)
    vFDIAcce_DAD_f1_adv_std=np.std(vFDIAcce_DAD_f1_adv)
    ###############################################################################  
    ################################################################################    
    vFDISpeed_predictorONLY_precsionn=[]
    vFDISpeed_predictorONLY_recall=[]
    vFDISpeed_predictorONLY_f1=[]
    
    vFDISpeed_PCC_precsionn=[]
    vFDISpeed_PCC_recall=[]
    vFDISpeed_PCC_f1=[]
    
    vFDISpeed_DAD_precsionn=[]
    vFDISpeed_DAD_recall=[]
    vFDISpeed_DAD_f1=[]
    
    vFDISpeed_predictorONLY_precsionn_adv=[]
    vFDISpeed_predictorONLY_recall_adv=[]
    vFDISpeed_predictorONLY_f1_adv=[]
    
    vFDISpeed_PCC_precsionn_adv=[]
    vFDISpeed_PCC_recall_adv=[]
    vFDISpeed_PCC_f1_adv=[]
    
    vFDISpeed_DAD_precsionn_adv=[]
    vFDISpeed_DAD_recall_adv=[]
    vFDISpeed_DAD_f1_adv=[]
    for item in vFDISpeed:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeed_predictorONLY_precsionn.append(tmp['precision'])
            vFDISpeed_predictorONLY_recall.append(tmp['recall'])
            vFDISpeed_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeed_PCC_precsionn.append(tmp['precision'])
            vFDISpeed_PCC_recall.append(tmp['recall'])
            vFDISpeed_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeed_DAD_precsionn.append(tmp['precision'])
            vFDISpeed_DAD_recall.append(tmp['recall'])
            vFDISpeed_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeed_predictorONLY_precsionn_adv.append(tmp['precision'])
            vFDISpeed_predictorONLY_recall_adv.append(tmp['recall'])
            vFDISpeed_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeed_PCC_precsionn_adv.append(tmp['precision'])
            vFDISpeed_PCC_recall_adv.append(tmp['recall'])
            vFDISpeed_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeed_DAD_precsionn_adv.append(tmp['precision'])
            vFDISpeed_DAD_recall_adv.append(tmp['recall'])
            vFDISpeed_DAD_f1_adv.append(tmp['f1-score'])        
    vFDISpeed_predictorONLY_precsion_mean=np.mean(vFDISpeed_predictorONLY_precsionn)
    vFDISpeed_predictorONLY_recall_mean=np.mean(vFDISpeed_predictorONLY_recall)
    vFDISpeed_predictorONLY_f1_mean=np.mean(vFDISpeed_predictorONLY_f1)
    
    vFDISpeed_PCC_precsionn_mean=np.mean(vFDISpeed_PCC_precsionn)
    vFDISpeed_PCC_recall_mean=np.mean(vFDISpeed_PCC_recall)
    vFDISpeed_PCC_f1_mean=np.mean(vFDISpeed_PCC_f1)
    
    vFDISpeed_DAD_precsionn_mean=np.mean(vFDISpeed_DAD_precsionn)
    vFDISpeed_DAD_recall_mean=np.mean(vFDISpeed_DAD_recall)
    vFDISpeed_DAD_f1_mean=np.mean(vFDISpeed_DAD_f1)
    
    vFDISpeed_predictorONLY_precsionn_adv_mean=np.mean(vFDISpeed_predictorONLY_precsionn_adv)
    vFDISpeed_predictorONLY_recall_adv_mean=np.mean(vFDISpeed_predictorONLY_recall_adv)
    vFDISpeed_predictorONLY_f1_adv_mean=np.mean(vFDISpeed_predictorONLY_f1_adv)
    
    vFDISpeed_PCC_precsionn_adv_mean=np.mean(vFDISpeed_PCC_precsionn_adv)
    vFDISpeed_PCC_recall_adv_mean=np.mean(vFDISpeed_PCC_recall_adv)
    vFDISpeed_PCC_f1_adv_mean=np.mean(vFDISpeed_PCC_f1_adv)
    
    vFDISpeed_DAD_precsionn_adv_mean=np.mean(vFDISpeed_DAD_precsionn_adv)
    vFDISpeed_DAD_recall_adv_mean=np.mean(vFDISpeed_DAD_recall_adv)
    vFDISpeed_DAD_f1_adv_mean=np.mean(vFDISpeed_DAD_f1_adv)
    
    vFDISpeed_predictorONLY_precsion_std=np.std(vFDISpeed_predictorONLY_precsionn)
    vFDISpeed_predictorONLY_recall_std=np.std(vFDISpeed_predictorONLY_recall)
    vFDISpeed_predictorONLY_f1_std=np.std(vFDISpeed_predictorONLY_f1)
    
    vFDISpeed_PCC_precsionn_std=np.std(vFDISpeed_PCC_precsionn)
    vFDISpeed_PCC_recall_std=np.std(vFDISpeed_PCC_recall)
    vFDISpeed_PCC_f1_std=np.std(vFDISpeed_PCC_f1)
    
    vFDISpeed_DAD_precsionn_std=np.std(vFDISpeed_DAD_precsionn)
    vFDISpeed_DAD_recall_std=np.std(vFDISpeed_DAD_recall)
    vFDISpeed_DAD_f1_std=np.std(vFDISpeed_DAD_f1)
    
    vFDISpeed_predictorONLY_precsionn_adv_std=np.std(vFDISpeed_predictorONLY_precsionn_adv)
    vFDISpeed_predictorONLY_recall_adv_std=np.std(vFDISpeed_predictorONLY_recall_adv)
    vFDISpeed_predictorONLY_f1_adv_std=np.std(vFDISpeed_predictorONLY_f1_adv)
    
    vFDISpeed_PCC_precsionn_adv_std=np.std(vFDISpeed_PCC_precsionn_adv)
    vFDISpeed_PCC_recall_adv_std=np.std(vFDISpeed_PCC_recall_adv)
    vFDISpeed_PCC_f1_adv_std=np.std(vFDISpeed_PCC_f1_adv)
    
    vFDISpeed_DAD_precsionn_adv_std=np.std(vFDISpeed_DAD_precsionn_adv)
    vFDISpeed_DAD_recall_adv_std=np.std(vFDISpeed_DAD_recall_adv)
    vFDISpeed_DAD_f1_adv_std=np.std(vFDISpeed_DAD_f1_adv)
    ###############################################################################  
    ################################################################################    
    vFDILoc_predictorONLY_precsionn=[]
    vFDILoc_predictorONLY_recall=[]
    vFDILoc_predictorONLY_f1=[]
    
    vFDILoc_PCC_precsionn=[]
    vFDILoc_PCC_recall=[]
    vFDILoc_PCC_f1=[]
    
    vFDILoc_DAD_precsionn=[]
    vFDILoc_DAD_recall=[]
    vFDILoc_DAD_f1=[]
    
    vFDILoc_predictorONLY_precsionn_adv=[]
    vFDILoc_predictorONLY_recall_adv=[]
    vFDILoc_predictorONLY_f1_adv=[]
    
    vFDILoc_PCC_precsionn_adv=[]
    vFDILoc_PCC_recall_adv=[]
    vFDILoc_PCC_f1_adv=[]
    
    vFDILoc_DAD_precsionn_adv=[]
    vFDILoc_DAD_recall_adv=[]
    vFDILoc_DAD_f1_adv=[]
    for item in vFDILoc:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDILoc_predictorONLY_precsionn.append(tmp['precision'])
            vFDILoc_predictorONLY_recall.append(tmp['recall'])
            vFDILoc_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDILoc_PCC_precsionn.append(tmp['precision'])
            vFDILoc_PCC_recall.append(tmp['recall'])
            vFDILoc_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDILoc_DAD_precsionn.append(tmp['precision'])
            vFDILoc_DAD_recall.append(tmp['recall'])
            vFDILoc_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDILoc_predictorONLY_precsionn_adv.append(tmp['precision'])
            vFDILoc_predictorONLY_recall_adv.append(tmp['recall'])
            vFDILoc_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDILoc_PCC_precsionn_adv.append(tmp['precision'])
            vFDILoc_PCC_recall_adv.append(tmp['recall'])
            vFDILoc_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDILoc_DAD_precsionn_adv.append(tmp['precision'])
            vFDILoc_DAD_recall_adv.append(tmp['recall'])
            vFDILoc_DAD_f1_adv.append(tmp['f1-score'])        
    vFDILoc_predictorONLY_precsion_mean=np.mean(vFDILoc_predictorONLY_precsionn)
    vFDILoc_predictorONLY_recall_mean=np.mean(vFDILoc_predictorONLY_recall)
    vFDILoc_predictorONLY_f1_mean=np.mean(vFDILoc_predictorONLY_f1)
    
    vFDILoc_PCC_precsionn_mean=np.mean(vFDILoc_PCC_precsionn)
    vFDILoc_PCC_recall_mean=np.mean(vFDILoc_PCC_recall)
    vFDILoc_PCC_f1_mean=np.mean(vFDILoc_PCC_f1)
    
    vFDILoc_DAD_precsionn_mean=np.mean(vFDILoc_DAD_precsionn)
    vFDILoc_DAD_recall_mean=np.mean(vFDILoc_DAD_recall)
    vFDILoc_DAD_f1_mean=np.mean(vFDILoc_DAD_f1)
    
    vFDILoc_predictorONLY_precsionn_adv_mean=np.mean(vFDILoc_predictorONLY_precsionn_adv)
    vFDILoc_predictorONLY_recall_adv_mean=np.mean(vFDILoc_predictorONLY_recall_adv)
    vFDILoc_predictorONLY_f1_adv_mean=np.mean(vFDILoc_predictorONLY_f1_adv)
    
    vFDILoc_PCC_precsionn_adv_mean=np.mean(vFDILoc_PCC_precsionn_adv)
    vFDILoc_PCC_recall_adv_mean=np.mean(vFDILoc_PCC_recall_adv)
    vFDILoc_PCC_f1_adv_mean=np.mean(vFDILoc_PCC_f1_adv)
    
    vFDILoc_DAD_precsionn_adv_mean=np.mean(vFDILoc_DAD_precsionn_adv)
    vFDILoc_DAD_recall_adv_mean=np.mean(vFDILoc_DAD_recall_adv)
    vFDILoc_DAD_f1_adv_mean=np.mean(vFDILoc_DAD_f1_adv)
    
    vFDILoc_predictorONLY_precsion_std=np.std(vFDILoc_predictorONLY_precsionn)
    vFDILoc_predictorONLY_recall_std=np.std(vFDILoc_predictorONLY_recall)
    vFDILoc_predictorONLY_f1_std=np.std(vFDILoc_predictorONLY_f1)
    
    vFDILoc_PCC_precsionn_std=np.std(vFDILoc_PCC_precsionn)
    vFDILoc_PCC_recall_std=np.std(vFDILoc_PCC_recall)
    vFDILoc_PCC_f1_std=np.std(vFDILoc_PCC_f1)
    
    vFDILoc_DAD_precsionn_std=np.std(vFDILoc_DAD_precsionn)
    vFDILoc_DAD_recall_std=np.std(vFDILoc_DAD_recall)
    vFDILoc_DAD_f1_std=np.std(vFDILoc_DAD_f1)
    
    vFDILoc_predictorONLY_precsionn_adv_std=np.std(vFDILoc_predictorONLY_precsionn_adv)
    vFDILoc_predictorONLY_recall_adv_std=np.std(vFDILoc_predictorONLY_recall_adv)
    vFDILoc_predictorONLY_f1_adv_std=np.std(vFDILoc_predictorONLY_f1_adv)
    
    vFDILoc_PCC_precsionn_adv_std=np.std(vFDILoc_PCC_precsionn_adv)
    vFDILoc_PCC_recall_adv_std=np.std(vFDILoc_PCC_recall_adv)
    vFDILoc_PCC_f1_adv_std=np.std(vFDILoc_PCC_f1_adv)
    
    vFDILoc_DAD_precsionn_adv_std=np.std(vFDILoc_DAD_precsionn_adv)
    vFDILoc_DAD_recall_adv_std=np.std(vFDILoc_DAD_recall_adv)
    vFDILoc_DAD_f1_adv_std=np.std(vFDILoc_DAD_f1_adv)
    ###############################################################################  
    
    ################################################################################    
    vFDIAcceSpeed_predictorONLY_precsionn=[]
    vFDIAcceSpeed_predictorONLY_recall=[]
    vFDIAcceSpeed_predictorONLY_f1=[]
    
    vFDIAcceSpeed_PCC_precsionn=[]
    vFDIAcceSpeed_PCC_recall=[]
    vFDIAcceSpeed_PCC_f1=[]
    
    vFDIAcceSpeed_DAD_precsionn=[]
    vFDIAcceSpeed_DAD_recall=[]
    vFDIAcceSpeed_DAD_f1=[]
    
    vFDIAcceSpeed_predictorONLY_precsionn_adv=[]
    vFDIAcceSpeed_predictorONLY_recall_adv=[]
    vFDIAcceSpeed_predictorONLY_f1_adv=[]
    
    vFDIAcceSpeed_PCC_precsionn_adv=[]
    vFDIAcceSpeed_PCC_recall_adv=[]
    vFDIAcceSpeed_PCC_f1_adv=[]
    
    vFDIAcceSpeed_DAD_precsionn_adv=[]
    vFDIAcceSpeed_DAD_recall_adv=[]
    vFDIAcceSpeed_DAD_f1_adv=[]
    for item in vFDIAcceSpeed:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceSpeed_predictorONLY_precsionn.append(tmp['precision'])
            vFDIAcceSpeed_predictorONLY_recall.append(tmp['recall'])
            vFDIAcceSpeed_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceSpeed_PCC_precsionn.append(tmp['precision'])
            vFDIAcceSpeed_PCC_recall.append(tmp['recall'])
            vFDIAcceSpeed_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceSpeed_DAD_precsionn.append(tmp['precision'])
            vFDIAcceSpeed_DAD_recall.append(tmp['recall'])
            vFDIAcceSpeed_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceSpeed_predictorONLY_precsionn_adv.append(tmp['precision'])
            vFDIAcceSpeed_predictorONLY_recall_adv.append(tmp['recall'])
            vFDIAcceSpeed_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceSpeed_PCC_precsionn_adv.append(tmp['precision'])
            vFDIAcceSpeed_PCC_recall_adv.append(tmp['recall'])
            vFDIAcceSpeed_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceSpeed_DAD_precsionn_adv.append(tmp['precision'])
            vFDIAcceSpeed_DAD_recall_adv.append(tmp['recall'])
            vFDIAcceSpeed_DAD_f1_adv.append(tmp['f1-score'])        
    vFDIAcceSpeed_predictorONLY_precsion_mean=np.mean(vFDIAcceSpeed_predictorONLY_precsionn)
    vFDIAcceSpeed_predictorONLY_recall_mean=np.mean(vFDIAcceSpeed_predictorONLY_recall)
    vFDIAcceSpeed_predictorONLY_f1_mean=np.mean(vFDIAcceSpeed_predictorONLY_f1)
    
    vFDIAcceSpeed_PCC_precsionn_mean=np.mean(vFDIAcceSpeed_PCC_precsionn)
    vFDIAcceSpeed_PCC_recall_mean=np.mean(vFDIAcceSpeed_PCC_recall)
    vFDIAcceSpeed_PCC_f1_mean=np.mean(vFDIAcceSpeed_PCC_f1)
    
    vFDIAcceSpeed_DAD_precsionn_mean=np.mean(vFDIAcceSpeed_DAD_precsionn)
    vFDIAcceSpeed_DAD_recall_mean=np.mean(vFDIAcceSpeed_DAD_recall)
    vFDIAcceSpeed_DAD_f1_mean=np.mean(vFDIAcceSpeed_DAD_f1)
    
    vFDIAcceSpeed_predictorONLY_precsionn_adv_mean=np.mean(vFDIAcceSpeed_predictorONLY_precsionn_adv)
    vFDIAcceSpeed_predictorONLY_recall_adv_mean=np.mean(vFDIAcceSpeed_predictorONLY_recall_adv)
    vFDIAcceSpeed_predictorONLY_f1_adv_mean=np.mean(vFDIAcceSpeed_predictorONLY_f1_adv)
    
    vFDIAcceSpeed_PCC_precsionn_adv_mean=np.mean(vFDIAcceSpeed_PCC_precsionn_adv)
    vFDIAcceSpeed_PCC_recall_adv_mean=np.mean(vFDIAcceSpeed_PCC_recall_adv)
    vFDIAcceSpeed_PCC_f1_adv_mean=np.mean(vFDIAcceSpeed_PCC_f1_adv)
    
    vFDIAcceSpeed_DAD_precsionn_adv_mean=np.mean(vFDIAcceSpeed_DAD_precsionn_adv)
    vFDIAcceSpeed_DAD_recall_adv_mean=np.mean(vFDIAcceSpeed_DAD_recall_adv)
    vFDIAcceSpeed_DAD_f1_adv_mean=np.mean(vFDIAcceSpeed_DAD_f1_adv)
    
    vFDIAcceSpeed_predictorONLY_precsion_std=np.std(vFDIAcceSpeed_predictorONLY_precsionn)
    vFDIAcceSpeed_predictorONLY_recall_std=np.std(vFDIAcceSpeed_predictorONLY_recall)
    vFDIAcceSpeed_predictorONLY_f1_std=np.std(vFDIAcceSpeed_predictorONLY_f1)
    
    vFDIAcceSpeed_PCC_precsionn_std=np.std(vFDIAcceSpeed_PCC_precsionn)
    vFDIAcceSpeed_PCC_recall_std=np.std(vFDIAcceSpeed_PCC_recall)
    vFDIAcceSpeed_PCC_f1_std=np.std(vFDIAcceSpeed_PCC_f1)
    
    vFDIAcceSpeed_DAD_precsionn_std=np.std(vFDIAcceSpeed_DAD_precsionn)
    vFDIAcceSpeed_DAD_recall_std=np.std(vFDIAcceSpeed_DAD_recall)
    vFDIAcceSpeed_DAD_f1_std=np.std(vFDIAcceSpeed_DAD_f1)
    
    vFDIAcceSpeed_predictorONLY_precsionn_adv_std=np.std(vFDIAcceSpeed_predictorONLY_precsionn_adv)
    vFDIAcceSpeed_predictorONLY_recall_adv_std=np.std(vFDIAcceSpeed_predictorONLY_recall_adv)
    vFDIAcceSpeed_predictorONLY_f1_adv_std=np.std(vFDIAcceSpeed_predictorONLY_f1_adv)
    
    vFDIAcceSpeed_PCC_precsionn_adv_std=np.std(vFDIAcceSpeed_PCC_precsionn_adv)
    vFDIAcceSpeed_PCC_recall_adv_std=np.std(vFDIAcceSpeed_PCC_recall_adv)
    vFDIAcceSpeed_PCC_f1_adv_std=np.std(vFDIAcceSpeed_PCC_f1_adv)
    
    vFDIAcceSpeed_DAD_precsionn_adv_std=np.std(vFDIAcceSpeed_DAD_precsionn_adv)
    vFDIAcceSpeed_DAD_recall_adv_std=np.std(vFDIAcceSpeed_DAD_recall_adv)
    vFDIAcceSpeed_DAD_f1_adv_std=np.std(vFDIAcceSpeed_DAD_f1_adv)
    ###############################################################################  
    
    ################################################################################    
    vFDIAcceLoc_predictorONLY_precsionn=[]
    vFDIAcceLoc_predictorONLY_recall=[]
    vFDIAcceLoc_predictorONLY_f1=[]
    
    vFDIAcceLoc_PCC_precsionn=[]
    vFDIAcceLoc_PCC_recall=[]
    vFDIAcceLoc_PCC_f1=[]
    
    vFDIAcceLoc_DAD_precsionn=[]
    vFDIAcceLoc_DAD_recall=[]
    vFDIAcceLoc_DAD_f1=[]
    
    vFDIAcceLoc_predictorONLY_precsionn_adv=[]
    vFDIAcceLoc_predictorONLY_recall_adv=[]
    vFDIAcceLoc_predictorONLY_f1_adv=[]
    
    vFDIAcceLoc_PCC_precsionn_adv=[]
    vFDIAcceLoc_PCC_recall_adv=[]
    vFDIAcceLoc_PCC_f1_adv=[]
    
    vFDIAcceLoc_DAD_precsionn_adv=[]
    vFDIAcceLoc_DAD_recall_adv=[]
    vFDIAcceLoc_DAD_f1_adv=[]
    for item in vFDIAcceLoc:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceLoc_predictorONLY_precsionn.append(tmp['precision'])
            vFDIAcceLoc_predictorONLY_recall.append(tmp['recall'])
            vFDIAcceLoc_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceLoc_PCC_precsionn.append(tmp['precision'])
            vFDIAcceLoc_PCC_recall.append(tmp['recall'])
            vFDIAcceLoc_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceLoc_DAD_precsionn.append(tmp['precision'])
            vFDIAcceLoc_DAD_recall.append(tmp['recall'])
            vFDIAcceLoc_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceLoc_predictorONLY_precsionn_adv.append(tmp['precision'])
            vFDIAcceLoc_predictorONLY_recall_adv.append(tmp['recall'])
            vFDIAcceLoc_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceLoc_PCC_precsionn_adv.append(tmp['precision'])
            vFDIAcceLoc_PCC_recall_adv.append(tmp['recall'])
            vFDIAcceLoc_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAcceLoc_DAD_precsionn_adv.append(tmp['precision'])
            vFDIAcceLoc_DAD_recall_adv.append(tmp['recall'])
            vFDIAcceLoc_DAD_f1_adv.append(tmp['f1-score'])        
    vFDIAcceLoc_predictorONLY_precsion_mean=np.mean(vFDIAcceLoc_predictorONLY_precsionn)
    vFDIAcceLoc_predictorONLY_recall_mean=np.mean(vFDIAcceLoc_predictorONLY_recall)
    vFDIAcceLoc_predictorONLY_f1_mean=np.mean(vFDIAcceLoc_predictorONLY_f1)
    
    vFDIAcceLoc_PCC_precsionn_mean=np.mean(vFDIAcceLoc_PCC_precsionn)
    vFDIAcceLoc_PCC_recall_mean=np.mean(vFDIAcceLoc_PCC_recall)
    vFDIAcceLoc_PCC_f1_mean=np.mean(vFDIAcceLoc_PCC_f1)
    
    vFDIAcceLoc_DAD_precsionn_mean=np.mean(vFDIAcceLoc_DAD_precsionn)
    vFDIAcceLoc_DAD_recall_mean=np.mean(vFDIAcceLoc_DAD_recall)
    vFDIAcceLoc_DAD_f1_mean=np.mean(vFDIAcceLoc_DAD_f1)
    
    vFDIAcceLoc_predictorONLY_precsionn_adv_mean=np.mean(vFDIAcceLoc_predictorONLY_precsionn_adv)
    vFDIAcceLoc_predictorONLY_recall_adv_mean=np.mean(vFDIAcceLoc_predictorONLY_recall_adv)
    vFDIAcceLoc_predictorONLY_f1_adv_mean=np.mean(vFDIAcceLoc_predictorONLY_f1_adv)
    
    vFDIAcceLoc_PCC_precsionn_adv_mean=np.mean(vFDIAcceLoc_PCC_precsionn_adv)
    vFDIAcceLoc_PCC_recall_adv_mean=np.mean(vFDIAcceLoc_PCC_recall_adv)
    vFDIAcceLoc_PCC_f1_adv_mean=np.mean(vFDIAcceLoc_PCC_f1_adv)
    
    vFDIAcceLoc_DAD_precsionn_adv_mean=np.mean(vFDIAcceLoc_DAD_precsionn_adv)
    vFDIAcceLoc_DAD_recall_adv_mean=np.mean(vFDIAcceLoc_DAD_recall_adv)
    vFDIAcceLoc_DAD_f1_adv_mean=np.mean(vFDIAcceLoc_DAD_f1_adv)
    
    vFDIAcceLoc_predictorONLY_precsion_std=np.std(vFDIAcceLoc_predictorONLY_precsionn)
    vFDIAcceLoc_predictorONLY_recall_std=np.std(vFDIAcceLoc_predictorONLY_recall)
    vFDIAcceLoc_predictorONLY_f1_std=np.std(vFDIAcceLoc_predictorONLY_f1)
    
    vFDIAcceLoc_PCC_precsionn_std=np.std(vFDIAcceLoc_PCC_precsionn)
    vFDIAcceLoc_PCC_recall_std=np.std(vFDIAcceLoc_PCC_recall)
    vFDIAcceLoc_PCC_f1_std=np.std(vFDIAcceLoc_PCC_f1)
    
    vFDIAcceLoc_DAD_precsionn_std=np.std(vFDIAcceLoc_DAD_precsionn)
    vFDIAcceLoc_DAD_recall_std=np.std(vFDIAcceLoc_DAD_recall)
    vFDIAcceLoc_DAD_f1_std=np.std(vFDIAcceLoc_DAD_f1)
    
    vFDIAcceLoc_predictorONLY_precsionn_adv_std=np.std(vFDIAcceLoc_predictorONLY_precsionn_adv)
    vFDIAcceLoc_predictorONLY_recall_adv_std=np.std(vFDIAcceLoc_predictorONLY_recall_adv)
    vFDIAcceLoc_predictorONLY_f1_adv_std=np.std(vFDIAcceLoc_predictorONLY_f1_adv)
    
    vFDIAcceLoc_PCC_precsionn_adv_std=np.std(vFDIAcceLoc_PCC_precsionn_adv)
    vFDIAcceLoc_PCC_recall_adv_std=np.std(vFDIAcceLoc_PCC_recall_adv)
    vFDIAcceLoc_PCC_f1_adv_std=np.std(vFDIAcceLoc_PCC_f1_adv)
    
    vFDIAcceLoc_DAD_precsionn_adv_std=np.std(vFDIAcceLoc_DAD_precsionn_adv)
    vFDIAcceLoc_DAD_recall_adv_std=np.std(vFDIAcceLoc_DAD_recall_adv)
    vFDIAcceLoc_DAD_f1_adv_std=np.std(vFDIAcceLoc_DAD_f1_adv)
    ###############################################################################
    
    ################################################################################    
    vFDISpeedLoc_predictorONLY_precsionn=[]
    vFDISpeedLoc_predictorONLY_recall=[]
    vFDISpeedLoc_predictorONLY_f1=[]
    
    vFDISpeedLoc_PCC_precsionn=[]
    vFDISpeedLoc_PCC_recall=[]
    vFDISpeedLoc_PCC_f1=[]
    
    vFDISpeedLoc_DAD_precsionn=[]
    vFDISpeedLoc_DAD_recall=[]
    vFDISpeedLoc_DAD_f1=[]
    
    vFDISpeedLoc_predictorONLY_precsionn_adv=[]
    vFDISpeedLoc_predictorONLY_recall_adv=[]
    vFDISpeedLoc_predictorONLY_f1_adv=[]
    
    vFDISpeedLoc_PCC_precsionn_adv=[]
    vFDISpeedLoc_PCC_recall_adv=[]
    vFDISpeedLoc_PCC_f1_adv=[]
    
    vFDISpeedLoc_DAD_precsionn_adv=[]
    vFDISpeedLoc_DAD_recall_adv=[]
    vFDISpeedLoc_DAD_f1_adv=[]
    for item in vFDISpeedLoc:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeedLoc_predictorONLY_precsionn.append(tmp['precision'])
            vFDISpeedLoc_predictorONLY_recall.append(tmp['recall'])
            vFDISpeedLoc_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeedLoc_PCC_precsionn.append(tmp['precision'])
            vFDISpeedLoc_PCC_recall.append(tmp['recall'])
            vFDISpeedLoc_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeedLoc_DAD_precsionn.append(tmp['precision'])
            vFDISpeedLoc_DAD_recall.append(tmp['recall'])
            vFDISpeedLoc_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeedLoc_predictorONLY_precsionn_adv.append(tmp['precision'])
            vFDISpeedLoc_predictorONLY_recall_adv.append(tmp['recall'])
            vFDISpeedLoc_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeedLoc_PCC_precsionn_adv.append(tmp['precision'])
            vFDISpeedLoc_PCC_recall_adv.append(tmp['recall'])
            vFDISpeedLoc_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDISpeedLoc_DAD_precsionn_adv.append(tmp['precision'])
            vFDISpeedLoc_DAD_recall_adv.append(tmp['recall'])
            vFDISpeedLoc_DAD_f1_adv.append(tmp['f1-score'])        
    vFDISpeedLoc_predictorONLY_precsion_mean=np.mean(vFDISpeedLoc_predictorONLY_precsionn)
    vFDISpeedLoc_predictorONLY_recall_mean=np.mean(vFDISpeedLoc_predictorONLY_recall)
    vFDISpeedLoc_predictorONLY_f1_mean=np.mean(vFDISpeedLoc_predictorONLY_f1)
    
    vFDISpeedLoc_PCC_precsionn_mean=np.mean(vFDISpeedLoc_PCC_precsionn)
    vFDISpeedLoc_PCC_recall_mean=np.mean(vFDISpeedLoc_PCC_recall)
    vFDISpeedLoc_PCC_f1_mean=np.mean(vFDISpeedLoc_PCC_f1)
    
    vFDISpeedLoc_DAD_precsionn_mean=np.mean(vFDISpeedLoc_DAD_precsionn)
    vFDISpeedLoc_DAD_recall_mean=np.mean(vFDISpeedLoc_DAD_recall)
    vFDISpeedLoc_DAD_f1_mean=np.mean(vFDISpeedLoc_DAD_f1)
    
    vFDISpeedLoc_predictorONLY_precsionn_adv_mean=np.mean(vFDISpeedLoc_predictorONLY_precsionn_adv)
    vFDISpeedLoc_predictorONLY_recall_adv_mean=np.mean(vFDISpeedLoc_predictorONLY_recall_adv)
    vFDISpeedLoc_predictorONLY_f1_adv_mean=np.mean(vFDISpeedLoc_predictorONLY_f1_adv)
    
    vFDISpeedLoc_PCC_precsionn_adv_mean=np.mean(vFDISpeedLoc_PCC_precsionn_adv)
    vFDISpeedLoc_PCC_recall_adv_mean=np.mean(vFDISpeedLoc_PCC_recall_adv)
    vFDISpeedLoc_PCC_f1_adv_mean=np.mean(vFDISpeedLoc_PCC_f1_adv)
    
    vFDISpeedLoc_DAD_precsionn_adv_mean=np.mean(vFDISpeedLoc_DAD_precsionn_adv)
    vFDISpeedLoc_DAD_recall_adv_mean=np.mean(vFDISpeedLoc_DAD_recall_adv)
    vFDISpeedLoc_DAD_f1_adv_mean=np.mean(vFDISpeedLoc_DAD_f1_adv)
    
    vFDISpeedLoc_predictorONLY_precsion_std=np.std(vFDISpeedLoc_predictorONLY_precsionn)
    vFDISpeedLoc_predictorONLY_recall_std=np.std(vFDISpeedLoc_predictorONLY_recall)
    vFDISpeedLoc_predictorONLY_f1_std=np.std(vFDISpeedLoc_predictorONLY_f1)
    
    vFDISpeedLoc_PCC_precsionn_std=np.std(vFDISpeedLoc_PCC_precsionn)
    vFDISpeedLoc_PCC_recall_std=np.std(vFDISpeedLoc_PCC_recall)
    vFDISpeedLoc_PCC_f1_std=np.std(vFDISpeedLoc_PCC_f1)
    
    vFDISpeedLoc_DAD_precsionn_std=np.std(vFDISpeedLoc_DAD_precsionn)
    vFDISpeedLoc_DAD_recall_std=np.std(vFDISpeedLoc_DAD_recall)
    vFDISpeedLoc_DAD_f1_std=np.std(vFDISpeedLoc_DAD_f1)
    
    vFDISpeedLoc_predictorONLY_precsionn_adv_std=np.std(vFDISpeedLoc_predictorONLY_precsionn_adv)
    vFDISpeedLoc_predictorONLY_recall_adv_std=np.std(vFDISpeedLoc_predictorONLY_recall_adv)
    vFDISpeedLoc_predictorONLY_f1_adv_std=np.std(vFDISpeedLoc_predictorONLY_f1_adv)
    
    vFDISpeedLoc_PCC_precsionn_adv_std=np.std(vFDISpeedLoc_PCC_precsionn_adv)
    vFDISpeedLoc_PCC_recall_adv_std=np.std(vFDISpeedLoc_PCC_recall_adv)
    vFDISpeedLoc_PCC_f1_adv_std=np.std(vFDISpeedLoc_PCC_f1_adv)
    
    vFDISpeedLoc_DAD_precsionn_adv_std=np.std(vFDISpeedLoc_DAD_precsionn_adv)
    vFDISpeedLoc_DAD_recall_adv_std=np.std(vFDISpeedLoc_DAD_recall_adv)
    vFDISpeedLoc_DAD_f1_adv_std=np.std(vFDISpeedLoc_DAD_f1_adv)
    ###############################################################################
    
    ################################################################################    
    vFDIAll_predictorONLY_precsionn=[]
    vFDIAll_predictorONLY_recall=[]
    vFDIAll_predictorONLY_f1=[]
    
    vFDIAll_PCC_precsionn=[]
    vFDIAll_PCC_recall=[]
    vFDIAll_PCC_f1=[]
    
    vFDIAll_DAD_precsionn=[]
    vFDIAll_DAD_recall=[]
    vFDIAll_DAD_f1=[]
    
    vFDIAll_predictorONLY_precsionn_adv=[]
    vFDIAll_predictorONLY_recall_adv=[]
    vFDIAll_predictorONLY_f1_adv=[]
    
    vFDIAll_PCC_precsionn_adv=[]
    vFDIAll_PCC_recall_adv=[]
    vFDIAll_PCC_f1_adv=[]
    
    vFDIAll_DAD_precsionn_adv=[]
    vFDIAll_DAD_recall_adv=[]
    vFDIAll_DAD_f1_adv=[]
    for item in vFDIAll:
        if 'predictor only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAll_predictorONLY_precsionn.append(tmp['precision'])
            vFDIAll_predictorONLY_recall.append(tmp['recall'])
            vFDIAll_predictorONLY_f1.append(tmp['f1-score'])
        if 'physical consistency checker only' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAll_PCC_precsionn.append(tmp['precision'])
            vFDIAll_PCC_recall.append(tmp['recall'])
            vFDIAll_PCC_f1.append(tmp['f1-score'])
        if 'Proposed method' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAll_DAD_precsionn.append(tmp['precision'])
            vFDIAll_DAD_recall.append(tmp['recall'])
            vFDIAll_DAD_f1.append(tmp['f1-score'])
    
        if 'predictor only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAll_predictorONLY_precsionn_adv.append(tmp['precision'])
            vFDIAll_predictorONLY_recall_adv.append(tmp['recall'])
            vFDIAll_predictorONLY_f1_adv.append(tmp['f1-score'])
        if 'physical consistency checker only (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAll_PCC_precsionn_adv.append(tmp['precision'])
            vFDIAll_PCC_recall_adv.append(tmp['recall'])
            vFDIAll_PCC_f1_adv.append(tmp['f1-score'])
        if 'Proposed method (adv.)' in list(item['detection method']):
            tmp = ast.literal_eval(item['1'].values[0])
            
            vFDIAll_DAD_precsionn_adv.append(tmp['precision'])
            vFDIAll_DAD_recall_adv.append(tmp['recall'])
            vFDIAll_DAD_f1_adv.append(tmp['f1-score'])        
    vFDIAll_predictorONLY_precsion_mean=np.mean(vFDIAll_predictorONLY_precsionn)
    vFDIAll_predictorONLY_recall_mean=np.mean(vFDIAll_predictorONLY_recall)
    vFDIAll_predictorONLY_f1_mean=np.mean(vFDIAll_predictorONLY_f1)
    
    vFDIAll_PCC_precsionn_mean=np.mean(vFDIAll_PCC_precsionn)
    vFDIAll_PCC_recall_mean=np.mean(vFDIAll_PCC_recall)
    vFDIAll_PCC_f1_mean=np.mean(vFDIAll_PCC_f1)
    
    vFDIAll_DAD_precsionn_mean=np.mean(vFDIAll_DAD_precsionn)
    vFDIAll_DAD_recall_mean=np.mean(vFDIAll_DAD_recall)
    vFDIAll_DAD_f1_mean=np.mean(vFDIAll_DAD_f1)
    
    vFDIAll_predictorONLY_precsionn_adv_mean=np.mean(vFDIAll_predictorONLY_precsionn_adv)
    vFDIAll_predictorONLY_recall_adv_mean=np.mean(vFDIAll_predictorONLY_recall_adv)
    vFDIAll_predictorONLY_f1_adv_mean=np.mean(vFDIAll_predictorONLY_f1_adv)
    
    vFDIAll_PCC_precsionn_adv_mean=np.mean(vFDIAll_PCC_precsionn_adv)
    vFDIAll_PCC_recall_adv_mean=np.mean(vFDIAll_PCC_recall_adv)
    vFDIAll_PCC_f1_adv_mean=np.mean(vFDIAll_PCC_f1_adv)
    
    vFDIAll_DAD_precsionn_adv_mean=np.mean(vFDIAll_DAD_precsionn_adv)
    vFDIAll_DAD_recall_adv_mean=np.mean(vFDIAll_DAD_recall_adv)
    vFDIAll_DAD_f1_adv_mean=np.mean(vFDIAll_DAD_f1_adv)
    
    vFDIAll_predictorONLY_precsion_std=np.std(vFDIAll_predictorONLY_precsionn)
    vFDIAll_predictorONLY_recall_std=np.std(vFDIAll_predictorONLY_recall)
    vFDIAll_predictorONLY_f1_std=np.std(vFDIAll_predictorONLY_f1)
    
    vFDIAll_PCC_precsionn_std=np.std(vFDIAll_PCC_precsionn)
    vFDIAll_PCC_recall_std=np.std(vFDIAll_PCC_recall)
    vFDIAll_PCC_f1_std=np.std(vFDIAll_PCC_f1)
    
    vFDIAll_DAD_precsionn_std=np.std(vFDIAll_DAD_precsionn)
    vFDIAll_DAD_recall_std=np.std(vFDIAll_DAD_recall)
    vFDIAll_DAD_f1_std=np.std(vFDIAll_DAD_f1)
    
    vFDIAll_predictorONLY_precsionn_adv_std=np.std(vFDIAll_predictorONLY_precsionn_adv)
    vFDIAll_predictorONLY_recall_adv_std=np.std(vFDIAll_predictorONLY_recall_adv)
    vFDIAll_predictorONLY_f1_adv_std=np.std(vFDIAll_predictorONLY_f1_adv)
    
    vFDIAll_PCC_precsionn_adv_std=np.std(vFDIAll_PCC_precsionn_adv)
    vFDIAll_PCC_recall_adv_std=np.std(vFDIAll_PCC_recall_adv)
    vFDIAll_PCC_f1_adv_std=np.std(vFDIAll_PCC_f1_adv)
    
    vFDIAll_DAD_precsionn_adv_std=np.std(vFDIAll_DAD_precsionn_adv)
    vFDIAll_DAD_recall_adv_std=np.std(vFDIAll_DAD_recall_adv)
    vFDIAll_DAD_f1_adv_std=np.std(vFDIAll_DAD_f1_adv)
    
    if detector_type=='predictorONLY': 
        return mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean,mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std
    if detector_type=='PCC': 
        return mFDI_PCC_f1_mean,vFDIAcce_PCC_f1_mean,vFDISpeed_PCC_f1_mean,vFDILoc_PCC_f1_mean,vFDIAcceSpeed_PCC_f1_mean,vFDIAcceLoc_PCC_f1_mean,vFDISpeedLoc_PCC_f1_mean,vFDIAll_PCC_f1_mean,mFDI_PCC_f1_adv_mean,vFDIAcce_PCC_f1_adv_mean,vFDISpeed_PCC_f1_adv_mean,vFDILoc_PCC_f1_adv_mean,vFDIAcceSpeed_PCC_f1_adv_mean,vFDIAcceLoc_PCC_f1_adv_mean,vFDISpeedLoc_PCC_f1_adv_mean,vFDIAll_PCC_f1_adv_mean,mFDI_PCC_f1_std,vFDIAcce_PCC_f1_std,vFDISpeed_PCC_f1_std,vFDILoc_PCC_f1_std,vFDIAcceSpeed_PCC_f1_std,vFDIAcceLoc_PCC_f1_std,vFDISpeedLoc_PCC_f1_std,vFDIAll_PCC_f1_std,mFDI_PCC_f1_adv_std,vFDIAcce_PCC_f1_adv_std,vFDISpeed_PCC_f1_adv_std,vFDILoc_PCC_f1_adv_std,vFDIAcceSpeed_PCC_f1_adv_std,vFDIAcceLoc_PCC_f1_adv_std,vFDISpeedLoc_PCC_f1_adv_std,vFDIAll_PCC_f1_adv_std
   
    if detector_type=='DAD': 
        return mFDI_DAD_f1_mean,vFDIAcce_DAD_f1_mean,vFDISpeed_DAD_f1_mean,vFDILoc_DAD_f1_mean,vFDIAcceSpeed_DAD_f1_mean,vFDIAcceLoc_DAD_f1_mean,vFDISpeedLoc_DAD_f1_mean,vFDIAll_DAD_f1_mean,mFDI_DAD_f1_adv_mean,vFDIAcce_DAD_f1_adv_mean,vFDISpeed_DAD_f1_adv_mean,vFDILoc_DAD_f1_adv_mean,vFDIAcceSpeed_DAD_f1_adv_mean,vFDIAcceLoc_DAD_f1_adv_mean,vFDISpeedLoc_DAD_f1_adv_mean,vFDIAll_DAD_f1_adv_mean,mFDI_DAD_f1_std,vFDIAcce_DAD_f1_std,vFDISpeed_DAD_f1_std,vFDILoc_DAD_f1_std,vFDIAcceSpeed_DAD_f1_std,vFDIAcceLoc_DAD_f1_std,vFDISpeedLoc_DAD_f1_std,vFDIAll_DAD_f1_std,mFDI_DAD_f1_adv_std,vFDIAcce_DAD_f1_adv_std,vFDISpeed_DAD_f1_adv_std,vFDILoc_DAD_f1_adv_std,vFDIAcceSpeed_DAD_f1_adv_std,vFDIAcceLoc_DAD_f1_adv_std,vFDISpeedLoc_DAD_f1_adv_std,vFDIAll_DAD_f1_adv_std
   
# =============================================================================
x = np.array([0,1,2,3,4,5,6,7,8])
print('Analysing data from D1:LSTM...')
mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean,mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std = dataExtractor('Classification_report_2021-09-02_09.18.32_bsize512_nEpoch500_nWindow20_LSTM3_100.csv', 'predictorONLY')
LSTM_F1_mean = np.array([np.mean([mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean]),mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean])
LSTM_F1_adv_mean = np.array([np.mean([mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean]),mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean])
LSTM_F1_std = np.array([0,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std])
LSTM_F1_adv_std = np.array([0,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std])
print('Analysing data from D2:CNN...')
mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean,mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std = dataExtractor('Classification_report_2021-09-03_12.27.22_bsize512_nEpoch500_nWindow20_CNN.csv', 'predictorONLY')
CNN_F1_mean = np.array([np.mean([mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean]),mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean])
CNN_F1_adv_mean = np.array([np.mean([mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean]),mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean])
CNN_F1_std = np.array([0,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std])
CNN_F1_adv_std = np.array([0,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std])
print('Analysing data from D3:AE...')
mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean,mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std = dataExtractor('Classification_report_2021-09-03_01.27.03_bsize512_nEpoch500_nWindow20_convAE.csv', 'predictorONLY')
AE_F1_mean = np.array([np.mean([mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean]),mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean])
AE_F1_adv_mean = np.array([np.mean([mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean]),mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean])
AE_F1_std = np.array([0,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std])
AE_F1_adv_std = np.array([0,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std])
print('Analysing data from PCC...')
mFDI_PCC_f1_mean,vFDIAcce_PCC_f1_mean,vFDISpeed_PCC_f1_mean,vFDILoc_PCC_f1_mean,vFDIAcceSpeed_PCC_f1_mean,vFDIAcceLoc_PCC_f1_mean,vFDISpeedLoc_PCC_f1_mean,vFDIAll_PCC_f1_mean,mFDI_PCC_f1_adv_mean,vFDIAcce_PCC_f1_adv_mean,vFDISpeed_PCC_f1_adv_mean,vFDILoc_PCC_f1_adv_mean,vFDIAcceSpeed_PCC_f1_adv_mean,vFDIAcceLoc_PCC_f1_adv_mean,vFDISpeedLoc_PCC_f1_adv_mean,vFDIAll_PCC_f1_adv_mean,mFDI_PCC_f1_std,vFDIAcce_PCC_f1_std,vFDISpeed_PCC_f1_std,vFDILoc_PCC_f1_std,vFDIAcceSpeed_PCC_f1_std,vFDIAcceLoc_PCC_f1_std,vFDISpeedLoc_PCC_f1_std,vFDIAll_PCC_f1_std,mFDI_PCC_f1_adv_std,vFDIAcce_PCC_f1_adv_std,vFDISpeed_PCC_f1_adv_std,vFDILoc_PCC_f1_adv_std,vFDIAcceSpeed_PCC_f1_adv_std,vFDIAcceLoc_PCC_f1_adv_std,vFDISpeedLoc_PCC_f1_adv_std,vFDIAll_PCC_f1_adv_std = dataExtractor('Classification_report_2021-09-01_23.13.10_bsize512_nEpoch500_nWindow20_DAD.csv', 'PCC')
PCC_F1_mean = np.array([np.mean([mFDI_PCC_f1_mean,vFDIAcce_PCC_f1_mean,vFDISpeed_PCC_f1_mean,vFDILoc_PCC_f1_mean,vFDIAcceSpeed_PCC_f1_mean,vFDIAcceLoc_PCC_f1_mean,vFDISpeedLoc_PCC_f1_mean,vFDIAll_PCC_f1_mean]),mFDI_PCC_f1_mean,vFDIAcce_PCC_f1_mean,vFDISpeed_PCC_f1_mean,vFDILoc_PCC_f1_mean,vFDIAcceSpeed_PCC_f1_mean,vFDIAcceLoc_PCC_f1_mean,vFDISpeedLoc_PCC_f1_mean,vFDIAll_PCC_f1_mean])
PCC_F1_adv_mean = np.array([np.mean([mFDI_PCC_f1_adv_mean,vFDIAcce_PCC_f1_adv_mean,vFDISpeed_PCC_f1_adv_mean,vFDILoc_PCC_f1_adv_mean,vFDIAcceSpeed_PCC_f1_adv_mean,vFDIAcceLoc_PCC_f1_adv_mean,vFDISpeedLoc_PCC_f1_adv_mean,vFDIAll_PCC_f1_adv_mean]),mFDI_PCC_f1_adv_mean,vFDIAcce_PCC_f1_adv_mean,vFDISpeed_PCC_f1_adv_mean,vFDILoc_PCC_f1_adv_mean,vFDIAcceSpeed_PCC_f1_adv_mean,vFDIAcceLoc_PCC_f1_adv_mean,vFDISpeedLoc_PCC_f1_adv_mean,vFDIAll_PCC_f1_adv_mean])
PCC_F1_std = np.array([0,mFDI_PCC_f1_std,vFDIAcce_PCC_f1_std,vFDISpeed_PCC_f1_std,vFDILoc_PCC_f1_std,vFDIAcceSpeed_PCC_f1_std,vFDIAcceLoc_PCC_f1_std,vFDISpeedLoc_PCC_f1_std,vFDIAll_PCC_f1_std])
PCC_F1_adv_std = np.array([0,mFDI_PCC_f1_adv_std,vFDIAcce_PCC_f1_adv_std,vFDISpeed_PCC_f1_adv_std,vFDILoc_PCC_f1_adv_std,vFDIAcceSpeed_PCC_f1_adv_std,vFDIAcceLoc_PCC_f1_adv_std,vFDISpeedLoc_PCC_f1_adv_std,vFDIAll_PCC_f1_adv_std])
print('Analysing data from DAD...')
mFDI_DAD_f1_mean,vFDIAcce_DAD_f1_mean,vFDISpeed_DAD_f1_mean,vFDILoc_DAD_f1_mean,vFDIAcceSpeed_DAD_f1_mean,vFDIAcceLoc_DAD_f1_mean,vFDISpeedLoc_DAD_f1_mean,vFDIAll_DAD_f1_mean,mFDI_DAD_f1_adv_mean,vFDIAcce_DAD_f1_adv_mean,vFDISpeed_DAD_f1_adv_mean,vFDILoc_DAD_f1_adv_mean,vFDIAcceSpeed_DAD_f1_adv_mean,vFDIAcceLoc_DAD_f1_adv_mean,vFDISpeedLoc_DAD_f1_adv_mean,vFDIAll_DAD_f1_adv_mean,mFDI_DAD_f1_std,vFDIAcce_DAD_f1_std,vFDISpeed_DAD_f1_std,vFDILoc_DAD_f1_std,vFDIAcceSpeed_DAD_f1_std,vFDIAcceLoc_DAD_f1_std,vFDISpeedLoc_DAD_f1_std,vFDIAll_DAD_f1_std,mFDI_DAD_f1_adv_std,vFDIAcce_DAD_f1_adv_std,vFDISpeed_DAD_f1_adv_std,vFDILoc_DAD_f1_adv_std,vFDIAcceSpeed_DAD_f1_adv_std,vFDIAcceLoc_DAD_f1_adv_std,vFDISpeedLoc_DAD_f1_adv_std,vFDIAll_DAD_f1_adv_std = dataExtractor('Classification_report_2021-09-01_23.13.10_bsize512_nEpoch500_nWindow20_DAD.csv', 'DAD')
DAD_F1_mean = np.array([np.mean([mFDI_DAD_f1_mean,vFDIAcce_DAD_f1_mean,vFDISpeed_DAD_f1_mean,vFDILoc_DAD_f1_mean,vFDIAcceSpeed_DAD_f1_mean,vFDIAcceLoc_DAD_f1_mean,vFDISpeedLoc_DAD_f1_mean,vFDIAll_DAD_f1_mean]),mFDI_DAD_f1_mean,vFDIAcce_DAD_f1_mean,vFDISpeed_DAD_f1_mean,vFDILoc_DAD_f1_mean,vFDIAcceSpeed_DAD_f1_mean,vFDIAcceLoc_DAD_f1_mean,vFDISpeedLoc_DAD_f1_mean,vFDIAll_DAD_f1_mean])
DAD_F1_adv_mean = np.array([np.mean([mFDI_DAD_f1_adv_mean,vFDIAcce_DAD_f1_adv_mean,vFDISpeed_DAD_f1_adv_mean,vFDILoc_DAD_f1_adv_mean,vFDIAcceSpeed_DAD_f1_adv_mean,vFDIAcceLoc_DAD_f1_adv_mean,vFDISpeedLoc_DAD_f1_adv_mean,vFDIAll_DAD_f1_adv_mean]),mFDI_DAD_f1_adv_mean,vFDIAcce_DAD_f1_adv_mean,vFDISpeed_DAD_f1_adv_mean,vFDILoc_DAD_f1_adv_mean,vFDIAcceSpeed_DAD_f1_adv_mean,vFDIAcceLoc_DAD_f1_adv_mean,vFDISpeedLoc_DAD_f1_adv_mean,vFDIAll_DAD_f1_mean])
DAD_F1_std = np.array([0,mFDI_DAD_f1_std,vFDIAcce_DAD_f1_std,vFDISpeed_DAD_f1_std,vFDILoc_DAD_f1_std,vFDIAcceSpeed_DAD_f1_std,vFDIAcceLoc_DAD_f1_std,vFDISpeedLoc_DAD_f1_std,vFDIAll_DAD_f1_std])
DAD_F1_adv_std = np.array([0,mFDI_DAD_f1_adv_std,vFDIAcce_DAD_f1_adv_std,vFDISpeed_DAD_f1_adv_std,vFDILoc_DAD_f1_adv_std,vFDIAcceSpeed_DAD_f1_adv_std,vFDIAcceLoc_DAD_f1_adv_std,vFDISpeedLoc_DAD_f1_adv_std,vFDIAll_DAD_f1_std])


my_xticks = ['Average','m-FDI','v-FDI-Acce.','v-FDI-Speed','v-FDI-Loc.', 'v-FDI-\nAcce.Speed', 'v-FDI-\nAcce.Loc.', 'v-FDI-\nSpeed.Loc.', 'v-FDI-\nAcce.SpeedLoc.']
width = 0.19
fig, (ax1,ax2) = plt.subplots(2, sharex=True, tight_layout=True,figsize=(5,6))
# plt.ylim(0,1)
plt.xticks(x+width*1.5, my_xticks)

# ax1.bar(x, LSTM_F1,width,label='D1:LSTM',linestyle='-.')
# ax1.bar(x+width, CNN_F1,width,label='D2:CNN',linestyle='--')
# ax1.bar(x+width*2, PCC_F1,width,label='PCC',linestyle=':')
# ax1.bar(x+width*3, DAD_F1,width,label='DAD')

# ax2.bar(x, LSTM_F1_adv,width,label='LSTM (adv.)',linestyle='-.')
# ax2.bar(x+width, CNN_F1_adv,width,label='CNN (adv.)',linestyle='--')
# ax2.bar(x+width*2, PCC_F1_adv,width,label='PCC (adv.)',linestyle=':')
# ax2.bar(x+width*3, DAD_F1_adv,width,label='DAD (adv.)')
capsize = 8*width
alpha=0.5
ax1.bar(x, LSTM_F1_mean,yerr=LSTM_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D1:LSTM',edgecolor='white', color='tab:olive')
ax1.bar(x+width, CNN_F1_mean,yerr=CNN_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D2:CNN',edgecolor='white',color='tab:grey')
ax1.bar(x+2*width, AE_F1_mean,yerr=AE_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D3:AE',edgecolor='white',color='tab:purple')
# ax1.bar(x+width*2, PCC_F1,width=width,label='PCC',linestyle=':')
ax1.bar(x+width*3, DAD_F1_mean,yerr=DAD_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='Ours',edgecolor='white',color='tab:red')

ax2.bar(x, LSTM_F1_adv_mean,yerr=LSTM_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D1:LSTM',edgecolor='white',color='tab:olive')
ax2.bar(x+width, CNN_F1_adv_mean,yerr=CNN_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D2:CNN',edgecolor='white',color='tab:grey')
ax2.bar(x+2*width, AE_F1_adv_mean,yerr=AE_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D3:AE',edgecolor='white',color='tab:purple')
# ax2.bar(x+width*2, PCC_F1_adv,width=width,label='PCC (adv.)',linestyle=':')
ax2.bar(x+width*3, DAD_F1_adv_mean,yerr=DAD_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='DAD',edgecolor='white',color='tab:red')
for i in range(len(x)):
    LSTM_temp = LSTM_F1_adv_mean[i]
    CNN_temp = CNN_F1_adv_mean[i]
    AE_temp = AE_F1_adv_mean[i]
    if LSTM_temp == 0:
        ax2.scatter( x[i]+0*width,LSTM_temp,s=15,color='tab:olive')
    if CNN_temp == 0:
        ax2.scatter( x[i]+1*width,CNN_temp,s=15,color='tab:grey')
    if AE_temp == 0:
        ax2.scatter( x[i]+2*width,AE_temp,s=15,color='tab:purple')        


# plt.scatter( LSTM_F1_adv*-1,x,s=15,label='LSTM (adv.)',color='tab:olive')
# plt.scatter( CNN_F1_adv*-1,x+width,s=15,label='CNN (adv.)',color='tab:grey')
# # plt.scatter(PCC_F1_adv*-1,x+2*width,s=15,label='PCC (adv.)')
# # plt.scatter( DAD_F1_adv*-1,x+3*width,s=15,label='DAD (adv.)')
# plt.scatter( DAD_F1_adv*-1,x+2*width,s=15,label='DAD (adv.)',color='tab:red')

ax2.text(x[0]+3*width,(DAD_F1_adv_mean)[0], s=round( (DAD_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
ax2.text(x[0]+2*width,(AE_F1_adv_mean)[0], s=round( (AE_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)

ax2.text(x[0]+1*width,(CNN_F1_adv_mean)[0], s=round( (CNN_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
ax2.text(x[0]+0*width,(LSTM_F1_adv_mean)[0], s=round( (LSTM_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)

ax1.text(x[0]+3*width,(DAD_F1_mean)[0], s=round( (DAD_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
ax1.text(x[0]+2*width,(AE_F1_mean)[0], s=round( (AE_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)

ax1.text(x[0]+1*width,(CNN_F1_mean)[0], s=round( (CNN_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
ax1.text(x[0]+0*width,(LSTM_F1_mean)[0], s=round( (LSTM_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)


ax1.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.35))
# ax1.legend(ncol=1, loc='lower right')
# ax2.legend(ncol=1, loc='upper right')


ax1.set_title('Conventional Cyber-physical Attacks')
ax2.set_title('Adversarially-masked Cyber-physical Attacks')
ax1.set_ylabel('f-1 score')
ax2.set_ylabel('f-1 score')
plt.tick_params(axis='x', rotation=70)
# plt.ylim(0.0,1.0)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.savefig(os.path.join(save_path,'{}.pdf'.format('Evaluation_F1Score')),dpi=100,bbox_inches='tight')
plt.show()

##########################################
plt.figure()
plt.xticks(x+width*1.5, my_xticks)
plt.bar(x, LSTM_F1_mean,yerr=LSTM_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D1:LSTM',edgecolor='white', color='tab:olive')
plt.bar(x+width, CNN_F1_mean,yerr=CNN_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D2:CNN',edgecolor='white',color='tab:grey')
plt.bar(x+2*width, AE_F1_mean,yerr=AE_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D3:AE',edgecolor='white',color='tab:purple')
# ax1.bar(x+width*2, PCC_F1,width=width,label='PCC',linestyle=':')
plt.bar(x+width*3, DAD_F1_mean,yerr=DAD_F1_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='Ours',edgecolor='white',color='tab:red')
plt.text(x[0]+3*width,(DAD_F1_mean)[0], s=round( (DAD_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
plt.text(x[0]+2*width,(AE_F1_mean)[0], s=round( (AE_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)

plt.text(x[0]+1*width,(CNN_F1_mean)[0], s=round( (CNN_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
plt.text(x[0]+0*width,(LSTM_F1_mean)[0], s=round( (LSTM_F1_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
plt.legend()
plt.title('Conventional Cyber-physical Attacks')
plt.ylabel('f-1 score')
plt.tick_params(axis='x', rotation=70)
plt.savefig(os.path.join(save_path,'{}.pdf'.format('Evaluation_F1Score_split1')),dpi=100,bbox_inches='tight')
plt.show()
####################################
plt.figure()
plt.xticks(x+width*1.5, my_xticks)
plt.bar(x, LSTM_F1_adv_mean,yerr=LSTM_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D1:LSTM',edgecolor='white',color='tab:olive')
plt.bar(x+width, CNN_F1_adv_mean,yerr=CNN_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D2:CNN',edgecolor='white',color='tab:grey')
plt.bar(x+2*width, AE_F1_adv_mean,yerr=AE_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='D3:AE',edgecolor='white',color='tab:purple')
# ax2.bar(x+width*2, PCC_F1_adv,width=width,label='PCC (adv.)',linestyle=':')
plt.bar(x+width*3, DAD_F1_adv_mean,yerr=DAD_F1_adv_std, align='center', alpha=alpha, ecolor='black', capsize=capsize,width=width,label='DAD',edgecolor='white',color='tab:red')
for i in range(len(x)):
    LSTM_temp = LSTM_F1_adv_mean[i]
    CNN_temp = CNN_F1_adv_mean[i]
    AE_temp = AE_F1_adv_mean[i]
    if LSTM_temp == 0:
        plt.scatter( x[i]+0*width,LSTM_temp,s=15,color='tab:olive')
    if CNN_temp == 0:
        plt.scatter( x[i]+1*width,CNN_temp,s=15,color='tab:grey')
    if AE_temp == 0:
        plt.scatter( x[i]+2*width,AE_temp,s=15,color='tab:purple')    
plt.text(x[0]+3*width,(DAD_F1_adv_mean)[0], s=round( (DAD_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
plt.text(x[0]+2*width,(AE_F1_adv_mean)[0], s=round( (AE_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)

plt.text(x[0]+1*width,(CNN_F1_adv_mean)[0], s=round( (CNN_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
plt.text(x[0]+0*width,(LSTM_F1_adv_mean)[0], s=round( (LSTM_F1_adv_mean)[0],2), rotation='vertical',horizontalalignment= 'center', verticalalignment='top', fontsize=6.5)
plt.legend()
plt.title('Adversarially-masked Cyber-physical Attacks')
plt.ylabel('f-1 score')
plt.tick_params(axis='x', rotation=70)
plt.savefig(os.path.join(save_path,'{}.pdf'.format('Evaluation_F1Score_split2')),dpi=100,bbox_inches='tight')
plt.show()

# # Results for adversarial training 
# print('Results for adversarial training...')
# x = np.array([0,1,2,3,4,5,6,7,8])
# print('Analysing data from D1:LSTM...')
# mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean,mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std = dataExtractor('Classification_report_2021-09-05_09.12.32_bsize512_nEpoch500_nWindow20_LSTM3_100Adv.csv', 'predictorONLY')
# LSTM_F1_mean = np.array([np.mean([mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean]),mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean])
# LSTM_F1_adv_mean = np.array([np.mean([mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean]),mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean])
# LSTM_F1_std = np.array([0,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std])
# LSTM_F1_adv_std = np.array([0,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std])
# print('Analysing data from D2:CNN...')
# mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean,mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std = dataExtractor('Classification_report_2021-09-05_10.11.40_bsize512_nEpoch500_nWindow20_CNNAdv.csv', 'predictorONLY')
# CNN_F1_mean = np.array([np.mean([mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean]),mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean])
# CNN_F1_adv_mean = np.array([np.mean([mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean]),mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean])
# CNN_F1_std = np.array([0,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std])
# CNN_F1_adv_std = np.array([0,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std])
# print('Analysing data from D3:AE...')
# mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean,mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std = dataExtractor('Classification_report_2021-09-05_10.37.54_bsize512_nEpoch500_nWindow20_convAEAdv.csv', 'predictorONLY')
# AE_F1_mean = np.array([np.mean([mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean]),mFDI_predictorONLY_f1_mean,vFDIAcce_predictorONLY_f1_mean,vFDISpeed_predictorONLY_f1_mean,vFDILoc_predictorONLY_f1_mean,vFDIAcceSpeed_predictorONLY_f1_mean,vFDIAcceLoc_predictorONLY_f1_mean,vFDISpeedLoc_predictorONLY_f1_mean,vFDIAll_predictorONLY_f1_mean])
# AE_F1_adv_mean = np.array([np.mean([mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean]),mFDI_predictorONLY_f1_adv_mean,vFDIAcce_predictorONLY_f1_adv_mean,vFDISpeed_predictorONLY_f1_adv_mean,vFDILoc_predictorONLY_f1_adv_mean,vFDIAcceSpeed_predictorONLY_f1_adv_mean,vFDIAcceLoc_predictorONLY_f1_adv_mean,vFDISpeedLoc_predictorONLY_f1_adv_mean,vFDIAll_predictorONLY_f1_adv_mean])
# AE_F1_std = np.array([0,mFDI_predictorONLY_f1_std,vFDIAcce_predictorONLY_f1_std,vFDISpeed_predictorONLY_f1_std,vFDILoc_predictorONLY_f1_std,vFDIAcceSpeed_predictorONLY_f1_std,vFDIAcceLoc_predictorONLY_f1_std,vFDISpeedLoc_predictorONLY_f1_std,vFDIAll_predictorONLY_f1_std])
# AE_F1_adv_std = np.array([0,mFDI_predictorONLY_f1_adv_std,vFDIAcce_predictorONLY_f1_adv_std,vFDISpeed_predictorONLY_f1_adv_std,vFDILoc_predictorONLY_f1_adv_std,vFDIAcceSpeed_predictorONLY_f1_adv_std,vFDIAcceLoc_predictorONLY_f1_adv_std,vFDISpeedLoc_predictorONLY_f1_adv_std,vFDIAll_predictorONLY_f1_adv_std])
# print('Analysing data from PCC...')
# mFDI_PCC_f1_mean,vFDIAcce_PCC_f1_mean,vFDISpeed_PCC_f1_mean,vFDILoc_PCC_f1_mean,vFDIAcceSpeed_PCC_f1_mean,vFDIAcceLoc_PCC_f1_mean,vFDISpeedLoc_PCC_f1_mean,vFDIAll_PCC_f1_mean,mFDI_PCC_f1_adv_mean,vFDIAcce_PCC_f1_adv_mean,vFDISpeed_PCC_f1_adv_mean,vFDILoc_PCC_f1_adv_mean,vFDIAcceSpeed_PCC_f1_adv_mean,vFDIAcceLoc_PCC_f1_adv_mean,vFDISpeedLoc_PCC_f1_adv_mean,vFDIAll_PCC_f1_adv_mean,mFDI_PCC_f1_std,vFDIAcce_PCC_f1_std,vFDISpeed_PCC_f1_std,vFDILoc_PCC_f1_std,vFDIAcceSpeed_PCC_f1_std,vFDIAcceLoc_PCC_f1_std,vFDISpeedLoc_PCC_f1_std,vFDIAll_PCC_f1_std,mFDI_PCC_f1_adv_std,vFDIAcce_PCC_f1_adv_std,vFDISpeed_PCC_f1_adv_std,vFDILoc_PCC_f1_adv_std,vFDIAcceSpeed_PCC_f1_adv_std,vFDIAcceLoc_PCC_f1_adv_std,vFDISpeedLoc_PCC_f1_adv_std,vFDIAll_PCC_f1_adv_std = dataExtractor('Classification_report_2021-09-05_08.11.20_bsize512_nEpoch500_nWindow20_DADadv.csv', 'PCC')
# PCC_F1_mean = np.array([np.mean([mFDI_PCC_f1_mean,vFDIAcce_PCC_f1_mean,vFDISpeed_PCC_f1_mean,vFDILoc_PCC_f1_mean,vFDIAcceSpeed_PCC_f1_mean,vFDIAcceLoc_PCC_f1_mean,vFDISpeedLoc_PCC_f1_mean,vFDIAll_PCC_f1_mean]),mFDI_PCC_f1_mean,vFDIAcce_PCC_f1_mean,vFDISpeed_PCC_f1_mean,vFDILoc_PCC_f1_mean,vFDIAcceSpeed_PCC_f1_mean,vFDIAcceLoc_PCC_f1_mean,vFDISpeedLoc_PCC_f1_mean,vFDIAll_PCC_f1_mean])
# PCC_F1_adv_mean = np.array([np.mean([mFDI_PCC_f1_adv_mean,vFDIAcce_PCC_f1_adv_mean,vFDISpeed_PCC_f1_adv_mean,vFDILoc_PCC_f1_adv_mean,vFDIAcceSpeed_PCC_f1_adv_mean,vFDIAcceLoc_PCC_f1_adv_mean,vFDISpeedLoc_PCC_f1_adv_mean,vFDIAll_PCC_f1_adv_mean]),mFDI_PCC_f1_adv_mean,vFDIAcce_PCC_f1_adv_mean,vFDISpeed_PCC_f1_adv_mean,vFDILoc_PCC_f1_adv_mean,vFDIAcceSpeed_PCC_f1_adv_mean,vFDIAcceLoc_PCC_f1_adv_mean,vFDISpeedLoc_PCC_f1_adv_mean,vFDIAll_PCC_f1_adv_mean])
# PCC_F1_std = np.array([0,mFDI_PCC_f1_std,vFDIAcce_PCC_f1_std,vFDISpeed_PCC_f1_std,vFDILoc_PCC_f1_std,vFDIAcceSpeed_PCC_f1_std,vFDIAcceLoc_PCC_f1_std,vFDISpeedLoc_PCC_f1_std,vFDIAll_PCC_f1_std])
# PCC_F1_adv_std = np.array([0,mFDI_PCC_f1_adv_std,vFDIAcce_PCC_f1_adv_std,vFDISpeed_PCC_f1_adv_std,vFDILoc_PCC_f1_adv_std,vFDIAcceSpeed_PCC_f1_adv_std,vFDIAcceLoc_PCC_f1_adv_std,vFDISpeedLoc_PCC_f1_adv_std,vFDIAll_PCC_f1_adv_std])
# print('Analysing data from DAD...')
# mFDI_DAD_f1_mean,vFDIAcce_DAD_f1_mean,vFDISpeed_DAD_f1_mean,vFDILoc_DAD_f1_mean,vFDIAcceSpeed_DAD_f1_mean,vFDIAcceLoc_DAD_f1_mean,vFDISpeedLoc_DAD_f1_mean,vFDIAll_DAD_f1_mean,mFDI_DAD_f1_adv_mean,vFDIAcce_DAD_f1_adv_mean,vFDISpeed_DAD_f1_adv_mean,vFDILoc_DAD_f1_adv_mean,vFDIAcceSpeed_DAD_f1_adv_mean,vFDIAcceLoc_DAD_f1_adv_mean,vFDISpeedLoc_DAD_f1_adv_mean,vFDIAll_DAD_f1_adv_mean,mFDI_DAD_f1_std,vFDIAcce_DAD_f1_std,vFDISpeed_DAD_f1_std,vFDILoc_DAD_f1_std,vFDIAcceSpeed_DAD_f1_std,vFDIAcceLoc_DAD_f1_std,vFDISpeedLoc_DAD_f1_std,vFDIAll_DAD_f1_std,mFDI_DAD_f1_adv_std,vFDIAcce_DAD_f1_adv_std,vFDISpeed_DAD_f1_adv_std,vFDILoc_DAD_f1_adv_std,vFDIAcceSpeed_DAD_f1_adv_std,vFDIAcceLoc_DAD_f1_adv_std,vFDISpeedLoc_DAD_f1_adv_std,vFDIAll_DAD_f1_adv_std = dataExtractor('Classification_report_2021-09-05_08.11.20_bsize512_nEpoch500_nWindow20_DADadv.csv', 'DAD')
# DAD_F1_mean = np.array([np.mean([mFDI_DAD_f1_mean,vFDIAcce_DAD_f1_mean,vFDISpeed_DAD_f1_mean,vFDILoc_DAD_f1_mean,vFDIAcceSpeed_DAD_f1_mean,vFDIAcceLoc_DAD_f1_mean,vFDISpeedLoc_DAD_f1_mean,vFDIAll_DAD_f1_mean]),mFDI_DAD_f1_mean,vFDIAcce_DAD_f1_mean,vFDISpeed_DAD_f1_mean,vFDILoc_DAD_f1_mean,vFDIAcceSpeed_DAD_f1_mean,vFDIAcceLoc_DAD_f1_mean,vFDISpeedLoc_DAD_f1_mean,vFDIAll_DAD_f1_mean])
# DAD_F1_adv_mean = np.array([np.mean([mFDI_DAD_f1_adv_mean,vFDIAcce_DAD_f1_adv_mean,vFDISpeed_DAD_f1_adv_mean,vFDILoc_DAD_f1_adv_mean,vFDIAcceSpeed_DAD_f1_adv_mean,vFDIAcceLoc_DAD_f1_adv_mean,vFDISpeedLoc_DAD_f1_adv_mean,vFDIAll_DAD_f1_adv_mean]),mFDI_DAD_f1_adv_mean,vFDIAcce_DAD_f1_adv_mean,vFDISpeed_DAD_f1_adv_mean,vFDILoc_DAD_f1_adv_mean,vFDIAcceSpeed_DAD_f1_adv_mean,vFDIAcceLoc_DAD_f1_adv_mean,vFDISpeedLoc_DAD_f1_adv_mean,vFDIAll_DAD_f1_mean])
# DAD_F1_std = np.array([0,mFDI_DAD_f1_std,vFDIAcce_DAD_f1_std,vFDISpeed_DAD_f1_std,vFDILoc_DAD_f1_std,vFDIAcceSpeed_DAD_f1_std,vFDIAcceLoc_DAD_f1_std,vFDISpeedLoc_DAD_f1_std,vFDIAll_DAD_f1_std])
# DAD_F1_adv_std = np.array([0,mFDI_DAD_f1_adv_std,vFDIAcce_DAD_f1_adv_std,vFDISpeed_DAD_f1_adv_std,vFDILoc_DAD_f1_adv_std,vFDIAcceSpeed_DAD_f1_adv_std,vFDIAcceLoc_DAD_f1_adv_std,vFDISpeedLoc_DAD_f1_adv_std,vFDIAll_DAD_f1_std])
