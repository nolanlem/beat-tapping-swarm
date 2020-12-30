#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:33:39 2020

@author: nolanlem
"""
#%%

def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    avg_taps_per_bin = []
    for i, tap in enumerate(taps):
        try:
            num_taps_in_bin = len(taps[i])
            avg_taps_per_bin.append(num_taps_in_bin)

            if num_taps_in_bin > 1:   
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) # take first tap in window
            if num_taps_in_bin == 0:
                binnedtaps.append(np.nan)
            if num_taps_in_bin == 1:
                binnedtaps.append(taps[i][0])
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim


#%%
beatsegments = [(0,3), (3,6), (6,9)] # 6,3 beat sections

oscp_cond = {}
zcs_cond = {}

ptaps = {}

stim_ptaps = {}
for sync_conds, sync_str in zip(allconditions, sync_strs):
    print('working on', sync_str)
    oscp_cond[sync_str] = [] # for gen model, all zcs per coupling cond

    for sync_cond_version in sync_conds:
        sndbeatbin = sndbeatbins[sync_cond_version]
        #############################################
        ############### GEN MODEL ############
        #############################################
        oscp = []
        zcs_cond[sync_str] = [[] for elem in np.arange(len(sndbeatbin))]

        phases = np.load('../phases-zcs-old-exp/' + sync_cond_version + '.npy', allow_pickle=True)
        
        if sync_cond_version.startswith('p'):
            #phases = [np.where(zc > 0)[0] for zc in phases] # get indexes 
            phases = [list(librosa.samples_to_time(zc)) for zc in phases] # to seconds and list
        else:
            phases = [list(librosa.samples_to_time(zc)) for zc in phases]
        
        # bin each oscillators zcs according to beat bins of stim and append to list
        for p, osc in enumerate(phases):
            binned_zcs = binBeats(osc, sndbeatbin)
            binned_zcs, _ = binTapsFromBeatWindow(binned_zcs)    
            oscp.append(binned_zcs)
        
        dfmod = pd.DataFrame(oscp) # --> dataframe
        
        # interpolate / map positions within bb to circle map 
        for i in range(1, len(sndbeatbin)):
            zcstobin = dfmod.iloc[:,i-1].values
            binmin = sndbeatbin[i-1]
            binmax = sndbeatbin[i]
            binterp = interp1d([binmin, binmax], [0, 2*np.pi])
            zcs_cond[sync_str][i-1].extend(list(binterp(zcstobin)))                
        
        #############################################
        ############## SUBJECT TAPS ########
        #############################################
        ptaps[sync_str] = [[] for elem in np.arange(len(sndbeatbin))]
        
        stim_ptaps[sync_cond_version] = []
        for person in subject:  
            try:
                taps = subject_resps[person][sync_cond_version]
                #binned_taps = binBeats(tap_resps_secs, sndbeatbin)
                binned_taps = binBeats(taps, sndbeatbin)
                binned_taps, avg_taps_per_bin = binTapsFromBeatWindow(binned_taps) 
                stim_ptaps[sync_cond_version].append(binned_taps)  
            except:
                pass
            
        dfstim = pd.DataFrame(stim_ptaps[sync_cond_version])
        for i in range(1, len(sndbeatbin)):
            tapstobin = dfstim.iloc[:,i-1].values
            binmin = sndbeatbin[i-1]
            binmax = sndbeatbin[i]
            binterp = interp1d([binmin,binmax], [0,2*np.pi])
            ptaps[sync_str][i-1].extend(list(binterp(tapstobin)))
                

#%%
plt.figure()
fig_s, ax_s = plt.subplots(nrows=5, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(5,8), 
                            sharex=True)
# fig_m, ax_m = plt.subplots(nrows=5, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
#                             {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
#                             figsize=(5,8), 
#                             sharex=True)
fig_s.tight_layout()
# fig_m.tight_layout()


for ax_1, ax_2 in zip(ax_s.flat, ax_m.flat):
    ax_1.set_thetagrids([])
    ax_1.set_yticklabels([])
    ax_1.set_axisbelow(True)
    ax_1.grid(linewidth=0.1, alpha=1.0)
    # ax_2.set_thetagrids([])
    # ax_2.set_yticklabels([])
    # ax_2.set_axisbelow(True)
    # ax_2.grid(linewidth=0.1, alpha=1.0)

for sync_str, ax_ in zip(sync_strs, ax_s[:,1].flat):
    ax_.set_title(sync_str, fontsize=8)
# for sync_str, ax_ in zip(sync_strs, ax_m[:,1].flat):
#     ax_.set_title(sync_str)   
    
for i ,sync_str in enumerate(sync_strs):

    taps = pd.DataFrame(ptaps[sync_str]) # subject taps -> taps 
    zcs = pd.DataFrame(zcs_cond[sync_str]) # gen model zcs -> zcs
    
    # beatsection by beatsection
    for m, bw in enumerate(beatsegments):
        # SUBJECTS COP
        sbeat_col = taps.iloc[:,bw[0]:bw[1]].values.flatten() 
        R_s = np.abs(np.nanmean(np.exp(1j*sbeat_col)))
        psi_s = np.angle(np.nanmean(np.exp(1j*sbeat_col)))
        s_noise = np.random.random(len(sbeat_col))*0.3
        
        # GEN MODEL COP
        mbeat_col = zcs.iloc[bw[0]:bw[1],:].values.flatten() 
        R_m = np.abs(np.nanmean(np.exp(1j*mbeat_col)))
        psi_m = np.angle(np.nanmean(np.exp(1j*mbeat_col)))
        m_noise = np.random.random(len(mbeat_col))*0.3 
        
        # shift to zero center angle relative to gen model 
        if psi_m < np.pi:
            angdiff = np.pi - psi_m
            psi_m += angdiff
            mbeat_col += angdiff
            sbeat_col += angdiff
        if psi_m > np.pi:
            angdiff = psi_m - np.pi
            psi_m -= angdiff
            mbeat_col -= angdiff
            sbeat_col -= angdiff
        
        #ax_s[i,m].plot(np.arange(2), np.arange(2), alpha=0, color='white')
        ax_s[i,m].scatter(sbeat_col, 1-s_noise, s=12, alpha=0.8, c='firebrick', marker='.', edgecolors='none')
        ax_s[i,m].arrow(0, 0.0, psi_s, R_s, color='red', linewidth=0.8)

        ax_s[i,m].scatter(mbeat_col, 1-m_noise, s=12, alpha=0.8, c='blue', marker='.', edgecolors='none')
        ax_s[i,m].arrow(0, 0.0, psi_m, R_m, color='black', linewidth=0.8)

fig_s.savefig('./plots/bsa-old-exp/3-beat-segments/3-beat-seg-PCs.png', dpi=150)

#%%  
N = 40 # num osc
for sync_conds, sync_str in zip(allconditions, sync_strs):
    fig, ax = plt.subplots(nrows=len(sync_conds), ncols=1, figsize=(10,40), sharey=True, sharex=True)
    for ax_ in ax.flat:
        ax_.set_yticks([])
        #ax_.set_xticks()
    
    for n, sync_cond_version in enumerate(sync_conds):      
        
        y, _ = librosa.load('../allstims-old-exp/' + sync_cond_version + '.wav')
        phases = np.load('../phases-zcs-old-exp/' + sync_cond_version + '.npy', allow_pickle=True)
        sndbeatbin = sndbeatbins[sync_cond_version]
        
        ax[n].plot(y, linewidth=0.5) # plot wf
        ax[n].set_title(str(sync_cond_version), fontsize=6)
        
        p_vspacing = np.linspace(1, 4, N)
        vinc = (4-1)/N
        
        if sync_cond_version.startswith('p'):
            #phases = [np.where(zc > 0)[0] for zc in phases] # get indexes 
            phases = [list(librosa.samples_to_time(zc)) for zc in phases] # to seconds and list
            for i, p in enumerate(phases):
                ax[n].vlines(librosa.time_to_samples(p),p_vspacing[i],p_vspacing[i]+vinc, color='firebrick', linewidth=0.6)
        else:
            phases = [list(librosa.samples_to_time(zc)) for zc in phases]
            for i, p in enumerate(phases):
                ax[n].vlines(librosa.time_to_samples(p),p_vspacing[i],p_vspacing[i]+vinc, color='firebrick', linewidth=0.6)
        
        ax[n].vlines(librosa.time_to_samples(sndbeatbin), -2.5, 4, color='green', linewidth=0.8)    
                
        vspacing_l = np.linspace(-1, -2, len(subject))
        
        for i, person in enumerate(subject):
            try:
                taps = subject_resps[person][sync_cond_version]
                ax[n].vlines(librosa.time_to_samples(taps), vspacing_l[i], vspacing_l[i]-0.1, color='blue', linewidth=0.7, marker='^')
            except:
                pass
    fig.tight_layout()
    fig.savefig('./plots/raster-old-exp/' + sync_str + '.png', dpi=150)

    
#%%

        
        
        
        
        
        
        
        
        

