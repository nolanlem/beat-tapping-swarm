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




os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/')

thecsvfile = './psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/R_csv.csv'
df = pd.read_csv(thecsvfile)

conds = ['none', 'weak', 'medium', 'strong']
versions = ['t', 'n']
beatsegs = [str(elem) for elem in np.arange(6)]

fig_r, ax_r = plt.subplots(nrows=4, ncols=2, figsize=(10,5))
fig_psi, ax_psi = plt.subplots(nrows=4, ncols=2, figsize=(10,5))

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

        for rtmp, ptmp, b in zip(r_model, psi_model, np.arange(6)):
            R_m[cond][v][b] = rtmp
            psi_m[cond][v][b] = ptmp
            
        ax_r[i, j].plot(r_model, label='model', color='blue')
        #ax_r[i, j].plot(r_subject, label='subject')
        ax_r[i, j].set_title(v + ' ' + cond)
        
        #ax_psi[i, j].plot(np.radians(psi_subject) + np.pi)
        ax_psi[i, j].plot(np.radians(psi_model) + np.pi, color='blue')
        ax_psi[i, j].set_title(v + ' ' + cond)


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

csvdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/subject_csvs/'


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
#%% per subject scores, constraint: average iti score across beatsegs for strong > 0.67 
badsubjects = []
R_subj = {}
for csv in glob.glob(csvdir + '*.csv'):
    df = pd.read_csv(csv)
    for i, cond in enumerate(conds):
        for j,v in enumerate(versions):  
            
            r = df['R'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values#%%
            psi = df['psi'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values 
            #print(os.path.basename(csv), cond, v, np.nanmean(r), np.nanmean(psi)) 
            
            r_avg = np.nanmean(r)
            psi_avg = np.nanmean(psi)
            if cond == 'strong' and r_avg < 0.67:
                if os.path.basename(csv) in badsubjects:
                    pass 
                else:
                    badsubjects.append(os.path.basename(csv))
                #print(os.path.basename(csv), cond, v, round(np.nanmean(r),2))
#%%
#%% per subject scores, constraint: average iti score across beatsegs for weak > 0.4
#badsubjects = []
R_subj = {}
for csv in glob.glob(csvdir + '*.csv'):
    df = pd.read_csv(csv)
    for i, cond in enumerate(conds):
        for j,v in enumerate(versions):  
            
            r = df['R'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values#%%
            psi = df['psi'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values 
            #print(os.path.basename(csv), cond, v, np.nanmean(r), np.nanmean(psi)) 
            
            r_avg = np.nanmean(r)
            psi_avg = np.nanmean(psi)
            if cond == 'weak' and r_avg < 0.4:
                if os.path.basename(csv) in badsubjects:
                    pass 
                else:
                    print(cond, v, os.path.basename(csv), r_avg)
                    badsubjects.append(os.path.basename(csv))
                #print(os.path.basename(csv), cond, v, round(np.nanmean(r),2))
#%% check bad subjects data
## bad subjects (r < 0.67 on strong stim) 
for csv in badsubjects:
    df = pd.read_csv(csvdir + csv)
    print(csv)
    for i, cond in enumerate(conds):
        for j,v in enumerate(versions):         
            r = df['R'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values#%%
            psi = df['psi'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values 
            r_avg = np.nanmean(r)
            psi_avg = np.nanmean(psi)
            print(cond, v, r_avg)
    print('\n')
#%%
## all subjects 
for csv in glob.glob(csvdir + "*.csv"):
    df = pd.read_csv(csv)
    print(csv)
    for i, cond in enumerate(conds):
        for j,v in enumerate(versions):         
            r = df['R'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values#%%
            psi = df['psi'][(df["coupling condition"] == cond) & (df["timbre version"] == v)].values 
            r_avg = np.nanmean(r)
            psi_avg = np.nanmean(psi)
            print(cond, v, r_avg)
    print('\n')
#%%
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

#%%
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
            #psi = np.radians(psi)
            
            ax_r[i, j].plot(r, label='subject', alpha=0.4,linewidth=0.3)
            ax_r[i, j].set_title(v + ' ' + cond)
            
            psi = (psi + 2*np.pi)%(2*np.pi)
            ax_psi[i, j].plot(psi, linewidth=0.3, alpha=0.6)
            ax_psi[i, j].set_title(v + ' ' + cond)
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


#fig_r.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/subject-PC-plots/R-subjects.png', dpi=150)
#fig_psi.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/PCs/subject-PC-plots/psi-subjects.png', dpi=150)


#########################################################################################
#%%

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
            ax_comb[sc + vc, m].plot(np.arange(2), np.arange(2), alpha=0, color='red')
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
    

