#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:41:59 2020

@author: nolanlem
"""


#%% ######## NB: DON"T HAVE TO RUN....####
######### GET ITI OF STIMULI FROM GENERATIVE MODEL ##########

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
            if num_taps_in_bin > 0:            
                random_tap = np.random.randint(low=0, high=num_taps_in_bin)
                binnedtaps.append(taps[i][random_tap]) # take random tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim

sync_cond_phases = {}
iti_segment_mx, iti_segment_sx = {}, {}

for sync_str in sync_conds:
    sync_cond_phases[sync_str] = {}
    iti_segment_mx[sync_str] = {}
    iti_segment_sx[sync_str] = {}
    
    for v in t_strs:
        sync_cond_phases[sync_str][v] = [] 
        iti_segment_mx[sync_str][v] = []
        iti_segment_sx[sync_str][v] = []


for version, v in zip(all_timbre_conds, t_strs):
    for sync_cond, sync_str in zip(version, sync_conds):
        for sync_cond_version in sync_cond:
            y, _ = librosa.load('./allstims/' + sync_cond_version + '.wav')
            samp_len = len(y) 
            
            sndbeatbin = sndbeatbins[sync_cond_version]

            
            oscs = np.load('./all-stim-all-zcs/' + sync_cond_version + '.npy', allow_pickle=True)
            oscs_samps = [librosa.time_to_samples(osc) for osc in oscs]
            #oscs_cut = [osc[osc < samp_len] for osc in oscs_samps]          
            
            ## BEAT BINNING?
            # new_zcs = []
            # for osc in oscs_cut:
            #     binned_zcs = binBeats(osc, sndbeatbin)
            #     binned_zcs, _ = binTapsFromBeatWindow(binned_zcs)
            #     new_zcs.append(binned_zcs)
            
            #oscs_norm = [np.diff(osc)/librosa.time_to_samples(centerperiods[sync_cond_version]) for osc in new_zcs]
            
            oscs_norm = [np.diff(osc)/librosa.time_to_samples(centerperiods[sync_cond_version]) for osc in oscs_samps]
            sync_cond_phases[sync_str][v].extend(oscs_norm)
        
        #sync_cond_phases[sync_str][v] = pd.DataFrame(sync_cond_phases[sync_str][v])



beatsegments = [(0,3), (3,6), (6,9), (9,12), (12, 15), (15,18)] # 6,3 beat sections
beat_strs = [str(i) for i in range(len(beatsegments))]
beatsectionlabels = [str(elem[0]) + '-' + str(elem[1]) for elem in beatsegments] # form str array for plotting
xlabels_pos = np.arange(0,len(beatsegments))

for version, v in zip(all_timbre_conds, t_strs):
    for sync_str in sync_conds:
        #iti_mx_sync_cond[sync_str] = {}        
        #iti_mx_sync_cond[sync_str][v] = []

        #iti_sx_sync_cond[sync_str] = {}        
        #iti_sx_sync_cond[sync_str][v] = []    

        # load the zcs from the generative model 
        #zcs = np.load('./all-stim-all-zcs/' + sync_cond_version + '.npy', allow_pickle=True)
        zcs = sync_cond_phases[sync_str][v]
        
        #zcs_iti = np.diff(zcs_cut)/centerperiods[sync_cond_version]
        df_zcs = pd.DataFrame(zcs)
        df_zcs = df_zcs.replace(0, np.nan)
                
        for beatseg, beat_str in zip(beatsegments, beat_strs):
            zcs_col = df_zcs.iloc[:,beatseg[0]:beatseg[1]].values
            # get mean, std
            mx = np.nanmean(zcs_col)
            sx = np.nanmean(np.nanstd(zcs_col, axis=1))
       
            iti_segment_mx[sync_str][v].append(mx) 
            iti_segment_sx[sync_str][v].append(sx) 
        
        #iti_mx_sync_cond[sync_str][v].append(iti_segment_mx[sync_str][v])

#how many nans per columnbeat 
for sync_str in sync_conds:
    print('for ', sync_str, ':')
    df = pd.DataFrame(sync_cond_phases[sync_str]['n'])
    for i in range(18):
        col = df.values[:,i]          
        print('col ', i, ":", np.count_nonzero(np.isnan(col)))

for sync_str in sync_conds:
    print(sync_str, ' no timbre:')
    print(iti_segment_mx[sync_str]['n'])
    print(sync_str, ' timbre:')
    print(iti_segment_mx[sync_str]['t'])
### PLOT
plt.figure()
fig, ax = plt.subplots(2,2, sharex=True, figsize=(10,5), gridspec_kw={"height_ratios":[0.02,1]})

xrange = np.arange(0, len(beat_strs))

for i, sync_str in enumerate(sync_conds):
    #ax[1,0].plot(iti_segment_mx[sync_str]['n'])
    ax[1,1].errorbar(xrange, iti_segment_mx[sync_str]['n'], yerr=iti_segment_sx[sync_str]['n'], capsize=3,label=sync_str)   
    #ax[1,1].plot(iti_segment_mx[sync_str]['t'])
    ax[1,0].errorbar(xrange, iti_segment_mx[sync_str]['t'], yerr=iti_segment_sx[sync_str]['t'], capsize=3, label=sync_str)

cols = ['{}'.format(col) for col in ['Timbre', 'No Timbre']]
for ax_, col in zip(ax[0], cols):
    ax_.axis("off")
    ax_.set_title(col, fontweight='bold')

for ax_ in ax.flat:
    #ax_.set_ylim([-0.5, 3.1])
    ax_.set_ylim([0.7, 1.5])
    ax_.set_xticks(xlabels_pos)
    ax_.set_xticklabels(beatsectionlabels, fontsize=6)
    ax_.set_xlabel('beat segment')
        
plt.legend(title='coupling conditions', bbox_to_anchor=(1., 1.05))
fig.tight_layout()
plt.savefig('./analysis-scripts/plots/beat-segment-analysis/gen-model/iti-gen-model.png', dpi=130)
#%%
### NB: weak and medium don't appear to be that different in terms of average ITI per beat section  
### main difference is their sx sizes 
#### look below at difference between weak and medium 
print('mean across beat sections of SD:')
for v in t_strs:
    print('for ', v, ' weak:', np.mean(iti_segment_sx['weak'][v]), 'medium:', np.mean(iti_segment_sx['medium'][v]))
              
#%%  NB: don't have to run if not generating MODEL ITI....plot ITIs of all oscillators from gen model conds for t and n on same plot  
plt.figure()
fig, ax = plt.subplots(4,2,figsize=(10,10), sharex=True)

for n, v in enumerate(t_strs):
    for m, sync_str in enumerate(sync_conds):
        for osc in sync_cond_phases[sync_str][v]:
            ax[m,n].plot(osc, linewidth=0.4)
            ax[m,n].set_title(sync_str + ' ' + v)
            
for ax_ in ax.flat:
    ax_.set_ylim([0,7])
    ax_.set_xlim([0,100])           
plt.ylim([0.0, 7.5])
plt.xlim([0, 150])
plt.savefig('/Users/nolanlem/Desktop/iti-gen-model.png', dpi=160)