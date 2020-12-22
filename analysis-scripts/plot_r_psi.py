#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:07:02 2020

@author: nolanlem
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob 
import os 
import seaborn as sns 

sns.set()




os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/')

thecsvfile = './analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/R_csv.csv'
df = pd.read_csv(thecsvfile)

conds = ['none', 'weak', 'medium', 'strong']
versions = ['t', 'n']
beatsegs = [str(elem) for elem in np.arange(6)]

fig_r, ax_r = plt.subplots(nrows=4, ncols=2, figsize=(10,8), sharex=True, sharey=True)
fig_psi, ax_psi = plt.subplots(nrows=4, ncols=2, figsize=(10,8), sharex=True, sharey=True)

###### initialize SUBJECT R,psi dictionaries #########
R_m = {}
psi_m = {}
for i, cond in enumerate(conds):
    R_m[cond], psi_m[cond] = {}, {}
    for j, v in enumerate(versions):
        R_m[cond][v], psi_m[cond][v] = {}, {}
        for k, b in enumerate(beatsegs):
            R_m[cond][v][b], psi_m[cond][v][b] = [],[]

# plot model R, psi. save model params to R_m, psi_m for later
for i, cond in enumerate(conds):
    for j, v in enumerate(versions):
        r_model = df['R model'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values
        r_subject = df['R subject'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values
        r_diff = r_subject - r_model 
        
        
        psi_subject = df['psi subject'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values
        psi_model = df['psi model'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values
        psi_model = np.radians(psi_model)
 
        print(cond, v, r_model, psi_model)
        
        if v == 't':
            v_str = 'timbre'
        else:
            v_str = 'no timbre'

        for rtmp, ptmp, b in zip(r_model, psi_model, np.arange(6)):
            R_m[cond][v][b] = rtmp
            psi_m[cond][v][b] = ptmp
            
        ax_r[i, j].plot(r_model, label='model', color='blue')
        #ax_r[i, j].plot(r_subject, label='subject')
        ax_r[i, j].set_title(v_str + ' ' + cond)
        
        #ax_psi[i, j].plot(np.radians(psi_subject) + np.pi)
        ax_psi[i, j].plot(np.radians(psi_model) + np.pi, color='blue')
        ax_psi[i, j].set_title(v_str + ' ' + cond)


#fig_r.legend(title='coupling conditions', bbox_to_anchor=(1., 1.05))
# fig_r.suptitle('phase coherence magnitudes (R) per beat section')
# fig_psi.suptitle('phase coherence angle (psi) per beat section')
# plt.tight_layout()

# fig_r.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/R-plots.png', dpi=130)
# fig_psi.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/psi-plots.png', dpi=130)
  
                

conds = ['none', 'weak', 'medium', 'strong']
versions = ['t','n']
beatsections = [elem for elem in np.arange(6)]


# fig_r, ax_r = plt.subplots(nrows=4, ncols=2, figsize=(10,5))
# fig_psi, ax_psi = plt.subplots(nrows=4, ncols=2, figsize=(10,5))

csvdir = './analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/subject_csvs/'


###### initialize SUBJECT R,psi dictionaries #########
R_s, R_s_mx = {}, {}
psi_s, psi_s_mx = {}, {}
for i, cond in enumerate(conds):
    R_s[cond], psi_s[cond], R_s_mx[cond], psi_s_mx[cond] = {},{},{},{}
    for j,v in enumerate(versions):
        R_s[cond][v], psi_s[cond][v], R_s_mx[cond][v], psi_s_mx[cond][v] = {}, {}, {}, {}
        for k, b in enumerate(beatsections):
            R_s[cond][v][b], psi_s[cond][v][b] = [],[]
            R_s_mx[cond][v] = {}
            psi_s_mx[cond][v] = {}
#############################################


# get all subject R, psi from csv file 
for csv in glob.glob(csvdir + '*.csv'):
    df = pd.read_csv(csv)
    for i, cond in enumerate(conds):
        for j,v in enumerate(versions):
            for k, b in enumerate(beatsections):
                r = df['R'][(df["coupling condition"] == cond) & (df["timbre version"] == v) & (df["beat section"] == b)].values
                psi = df['psi'][(df["coupling condition"] == cond) & (df["timbre version"] == v) & (df["beat section"] == b)].values               
                R_s[cond][v][b].append(r[0])
                psi_s[cond][v][b].append(psi[0])

# take means within each beat section 
for i, cond in enumerate(conds):
    for j,v in enumerate(versions):
        for k, b in enumerate(beatsections):
            R_s_mx[cond][v][b] = np.nanmean(R_s[cond][v][b])
            psi_s_mx[cond][v][b] = np.nanmean(psi_s[cond][v][b])
          
for csv in glob.glob(csvdir + '*.csv'):
    print(csv)
    df = pd.read_csv(csv)
    
    for i, cond in enumerate(conds):
        for j,v in enumerate(versions):
            r = df['R'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values
            psi = df['psi'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values
            
            if v == 't':
                v_str = 'timbre'
            else:
                v_str = 'no-timbre'
            
            ax_r[i, j].plot(r, label='subject', alpha=0.4,linewidth=0.3)
            ax_r[i, j].set_title(v_str + ' ' + cond)
            
            psi = (psi + 2*np.pi)%(2*np.pi)
            ax_psi[i, j].plot(psi, linewidth=0.3, alpha=0.6)
            

            ax_psi[i, j].set_title(v_str + ' ' + cond)
            ax_psi[i, j].set_ylim([0, 2*np.pi])
            #ax_psi[i, j].hlines(np.pi, 0, 5, color='r', linewidth=0.5)
            
            # plot the averages 
            R_avg = np.array(list(R_s_mx[cond][v].items()))[:,1]  
            psi_avg = np.array(list(psi_s_mx[cond][v].items()))[:,1]
            #psi_avg = np.unwrap(psi_avg)
            #psi_avg = (psi_avg + 2*np.pi)%(2*np.pi)
            #psi_avg = psi_avg + np.pi
            
            ax_r[i, j].plot(R_avg, linewidth=0.5, alpha=0.5, color='red')
            ax_psi[i, j].plot(psi_avg, linewidth=0.5, alpha=0.5, color='red')


for ax_ in ax_r.flat:
    ax_.set_ylim([0,1.1])

for ax_ in ax_r[-1,:].flat:
    ax_.set_xticks(np.arange(6))
    ax_.set_xlabel('beat segment')
    xticklabs = ['1','2','3','4','5','6'] 
    ax_.set_xticklabels(xticklabs)



fig_r.tight_layout()
fig_psi.tight_layout()
fig_r.savefig('./analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/subject-PC-plots/R-subjects.png', dpi=150)
fig_psi.savefig('./analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/subject-PC-plots/psi-subjects.png', dpi=150)


#########################################################################################
#%% GENERATE/PLOT PCS CIRCLE MAPS PER BEAT SEGMENT ACROSS PARTICIPANTS 

versions = ['t', 'n']
plt.figure()
fig_comb, ax_comb = plt.subplots(nrows=8, ncols=len(beatsegs), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)

for ax_c in ax_comb.flat:
    ax_c.set_thetagrids([])
    ax_c.set_yticklabels([])
    ax_c.set_axisbelow(True)
    ax_c.grid(linewidth=0.1, alpha=1.0)




vc = 0
for j,v in enumerate(versions):  
    sc = 0
    for i, cond in enumerate(conds):
        for m in np.arange(6):
            #print(cond,  v, 'beat:', b, 'r model:', R_m[cond][v][m], 'psi model:', psi_m[cond][v][m])
            #print(cond,  v, 'beat:', b, 'r subj:', R_s_mx[cond][v][k], 'psi subj:', psi_s_mx[cond][v][k])
            #print(R_m[cond][v][m], psi_m[cond][v][m])
            ax_comb[sc + vc, m].plot(np.arange(2), np.arange(2), alpha=0, color='red') # have to have this phantom plot line for some reason
            #ax_comb[sc + vc, m].arrow(0,0,0,0.5, color='blue')

            ax_comb[sc + vc, m].plot(np.arange(2), np.arange(2), alpha=0, color='white')            
            ax_comb[sc + vc, m].arrow(0, 0.0, psi_m[cond][v][m], R_m[cond][v][m], color='blue', linewidth=1)
            ax_comb[sc + vc, m].arrow(0, 0.0, psi_s_mx[cond][v][m], R_s_mx[cond][v][m], color='firebrick', linewidth=1)
            
            print('[', sc + vc, ',', m, ']')
        sc += 1
    vc += 4

rowlabels = ['none', 'weak', 'medium', 'strong', 'none', 'weak', 'medium', 'strong']
for ax, row in zip(ax_comb[:,0], rowlabels):
    ax.set_ylabel(row, rotation=90, size='large', fontsize=8)
fig_comb.suptitle('subject vs. model phase coherence parameters')

fig_comb.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/subject-PC-plots/subject-model-PC-plot.png', dpi=150)
    
#%% count mturk workers in usable folder minus the badsubjects 
badsubjects = ['PARTICIPANT_kura-B1_2020-10-05_08h33.csv',
 'PARTICIPANT_kura-A2_2020-09-16_15h15.csv',
 'PARTICIPANT_kura-B1_2020-10-05_10h39.csv',
 'PARTICIPANT_kura-A1_2020-10-05_11h28.csv',
 'PARTICIPANT_kura-A2_2020-09-07_10h52.csv',
 'PARTICIPANT_kura-A2_2020-09-16_10h16.csv']
a1,a2,b1,b2 = [],[],[],[]
for csv in glob.glob('./psychopy/swarm-tapping-study/mturk-csv/usable-batch-11-8/*.csv'):
    csv_ = os.path.basename(csv).split('.')[0] + ".csv"
    if csv_ in badsubjects:
        pass 
    else:
        study = csv_.split('.')[0].split('_')[1].split('-')[1]
        if study == 'A1':
            a1.append(csv_)
        if study == 'A2':
            a2.append(csv_)
        if study == 'B1':
            b1.append(csv_)
        if study == 'B2':
            b2.append(csv_)
print('A1:', len(a1), 'A2:', len(a2), 'B1:', len(b1), 'B2:', len(b2))


#%%  parse subjects in usuable batch 
a1,a2,b1,b2 = [],[],[],[]
for csv in usable_subjects:
    csv_ = os.path.basename(csv).split('.')[0] + ".csv"
    study = csv_.split('.')[0].split('_')[1].split('-')[1]
    if study == 'A1':
        a1.append(csv_)
    if study == 'A2':
        a2.append(csv_)
    if study == 'B1':
        b1.append(csv_)
    if study == 'B2':
        b2.append(csv_)
        
print('A1:', len(a1), 'A2:', len(a2), 'B1:', len(b1), 'B2:', len(b2), 'total:', len(a1) + len(a2) + len(b1) + len(b2))

mturk_prompts = ['If you are a MTurk worker, what is your MTurk worker ID?', "If you are a MTurk worker, what is your MTurk worker ID?'"]
email_prompt = "If you are not an MTurk worker, please provide your email address" 
 

stanford_subjects, mturk_subjects = [], []

batch_folder = './mturk-csv/usable-batch-11-8/'
for csv_file in glob.glob(batch_folder + '*.csv'):
    if os.path.basename(csv_file).split('.')[0] not in badsubjects:
            
        csvbasename = os.path.basename(csv_file)        
        csv_data = pd.read_csv(csv_file, keep_default_na=False)
                
        mturk_id = 'none'
        email = 'none'
        
        try:
            mturk_id = csv_data[mturk_prompts[0]][0]
        except:
            pass    
        try:
            mturk_id = csv_data[mturk_prompts[1]][0]
        except:
            pass
        try:
            email = csv_data[email_prompt][0]
        except:
            pass
  
        if (mturk_id != 'none') and (mturk_id != ''):
            mturk_subjects.append(csvbasename)
            subjectplotdir = './analysis-scripts/plots/' + batch_folder + '/subjects/' + mturk_id        
        else:

            if (email != 'none') and (email != ''):
                #print('email given', email)
                stanford_subjects.append(csvbasename)
                subjectplotdir = './analysis-scripts/plots/' + batch_folder + '/subjects/' + email         
    
            else:
                inits = csv_data['Participant Initials'][0] 
                stanford_subjects.append(csvbasename)

                                 
print('stanford total:', len(stanford_subjects), 'mturk total:', len(mturk_subjects))

a1,a2,b1,b2 = [],[],[],[]
for csv in stanford_subjects:
    csv_ = os.path.basename(csv).split('.')[0] + ".csv"
    study = csv_.split('.')[0].split('_')[1].split('-')[1]
    if study == 'A1':
        a1.append(csv_)
    if study == 'A2':
        a2.append(csv_)
    if study == 'B1':
        b1.append(csv_)
    if study == 'B2':
        b2.append(csv_)
print('STANFORD TOTALS PER STUDY')
print('A1:', len(a1), 'A2:', len(a2), 'B1:', len(b1), 'B2:', len(b2), 'total:', len(a1) + len(a2) + len(b1) + len(b2))

a1,a2,b1,b2 = [],[],[],[]
for csv in mturk_subjects:
    csv_ = os.path.basename(csv).split('.')[0] + ".csv"
    study = csv_.split('.')[0].split('_')[1].split('-')[1]
    if study == 'A1':
        a1.append(csv_)
    if study == 'A2':
        a2.append(csv_)
    if study == 'B1':
        b1.append(csv_)
    if study == 'B2':
        b2.append(csv_)
print('MTURK TOTALS PER STUDY')
print('A1:', len(a1), 'A2:', len(a2), 'B1:', len(b1), 'B2:', len(b2), 'total:', len(a1) + len(a2) + len(b1) + len(b2))
