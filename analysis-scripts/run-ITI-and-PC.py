#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:38:35 2020

@author: nolanlem
"""


import numpy as np 
import pandas as pd
import glob 
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker
import sys
import os
from ast import literal_eval
from io import StringIO
import itertools
from scipy.signal import find_peaks
from collections import defaultdict
import librosa
from scipy.stats import sem
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.cm as cm
import datetime 
from collections import defaultdict
import scipy.stats
import csv


import seaborn as sns
sns.set()
os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/')


sr=22050


def removeStrFormatting(str_array):
    for str_arr in str_array:
        str_arr = str_arr[1:-1] # remove "'[" and "]'"
        str_arr = str.split(str_arr, ',') # split strings
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
        #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr


def formFixedBeatBins(wf, thesnd, limitpeaks=False):
    strsnd= os.path.basename(thesnd).split('.')[0]
    beat_bins = 0
    #amp_peaks, _ = find_peaks(wf, height=0.25, distance=sr/3.0) # get amplitude envelope and return peaks
    amp_peaks, _ = find_peaks(wf, height=0.4, distance=sr/2.0) # get amplitude envelope and return peaks
    #print 'numpks %r = %r'%(strsnd, len(amp_peaks))
    avg_int_bb = librosa.samples_to_time(np.average(np.diff(amp_peaks)))
    idealperiods[strsnd] = avg_int_bb     
    
    # get the ideal period from the sound file path
    idealperiod = 60./float(os.path.basename(snd).split('.')[0].split('_')[1])
    fixed_bb = [avg_int_bb*i for i in range(len(amp_peaks))]

    # for vweak case, amp env doesn't work very well 
    # none of the audio should have more than 15 'beats' therefore if find_peaks
    # returns too many peaks, use the 'idealperiod' to create the fixed beat window array
    if limitpeaks==True:
        if len(amp_peaks) > 15:
            print('too many amp peaks')
            fixed_bb = [idealperiod*i for i in range(14)]

    avg_bpm = 60./avg_int_bb # save the avg bpm depending on avg period
    beat_bins = librosa.samples_to_time(amp_peaks) # convert to samples

    # shift over fixed beat window depending on if its > or < first amplitude env peak 
    if fixed_bb[0] < beat_bins[0]:
        fixed_bb += (beat_bins[0] - fixed_bb[0])
    if fixed_bb[0] > beat_bins[0]:
        fixed_bb -= (fixed_bb[0] - beat_bins[0])
    # shift over half window (makes the "GT beat" at 180 deg in phase coherence plots )
    for i in range(1, len(fixed_bb)):
        fixed_bb[i-1] = fixed_bb[i-1] + (fixed_bb[i] - fixed_bb[i-1])/2 
    fixed_bb[-1] = fixed_bb[-1] + avg_int_bb/2. # last in array
    
    if fixed_bb[0] >= avg_int_bb:
        fixed_bb = np.insert(fixed_bb, 0, fixed_bb[0] - avg_int_bb)
    return fixed_bb, avg_bpm, amp_peaks

def flatten2DList(thelist):
    flatlist = list(itertools.chain(*thelist))
    return flatlist

def binBeats(taps, beat_bins):
    taps = np.array(taps)
    digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
    bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
    return bins

def binTapsFromBeatWindow(taps):
    binnedtaps = []
    for i, tap in enumerate(taps):
        try:
            binnedtaps.append(taps[i][0]) # take first tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    return binnedtaps

def removeStrFormatting(str_arr):
    str_arr = str_arr[1:-1] # remove "'[" and "]'"
    str_arr = str.split(str_arr, ',') # split strings
    try:
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
    except ValueError:
        pass
    #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr

def makeDir(dirname):
    if os.path.exists(dirname) == False:
        print('making directory: ', dirname)
        os.mkdir(dirname)

#%%
# A1 no(1,2)        timbre(1,2)
# B1 timbre(1,2)    no(1,2)
# A2 no(3,4)        timbre(3,4)
# B2 timbre(3,4)    no(3,4)
#
pp_dir_A1 = os.path.join('./psychopy','swarm-tapping-study','psychopy-A1')
pp_dir_A2 = os.path.join('./psychopy','swarm-tapping-study','psychopy-A2')
pp_dir_B1 = os.path.join('./psychopy','swarm-tapping-study','psychopy-B1')
pp_dir_B2 = os.path.join('./psychopy','swarm-tapping-study','psychopy-B2')

stim_block_1 = 'block_1/'
stim_block_2 = 'block_2/'

pp_stims = [pp_dir_A1, pp_dir_A2, pp_dir_B1, pp_dir_B2]
blocks = [stim_block_1, stim_block_2]


############## get sndfile strings #####################
sndfiles_no = []
sndfiles_timbre = []

# all the NO_TIMBRE sounds
for snd in glob.glob(os.path.join(pp_dir_A1, stim_block_1, "*.wav")):
    sndfiles_no.append(snd)
for snd in glob.glob(os.path.join(pp_dir_B1, stim_block_2, "*.wav")):
    sndfiles_no.append(snd)
for snd in glob.glob(os.path.join(pp_dir_A2, stim_block_1, "*.wav")):
    sndfiles_no.append(snd)
for snd in glob.glob(os.path.join(pp_dir_B2, stim_block_2, "*.wav")):
    sndfiles_no.append(snd)

# all the TIMBRE sounds
for snd in glob.glob(os.path.join(pp_dir_A1, stim_block_2, "*.wav")):
    sndfiles_timbre.append(snd)
for snd in glob.glob(os.path.join(pp_dir_B1, stim_block_1, "*.wav")):
    sndfiles_timbre.append(snd)
for snd in glob.glob(os.path.join(pp_dir_A2, stim_block_2, "*.wav")):
    sndfiles_timbre.append(snd)
for snd in glob.glob(os.path.join(pp_dir_B2, stim_block_1, "*.wav")):
    sndfiles_timbre.append(snd)


all_snds = [sndfiles_no, sndfiles_timbre]

timbre_conds = ['no', 'timbre']
sync_conds = ['none', 'weak', 'medium', 'strong']
sndfiles = {}

for i, timbre_cond in enumerate(timbre_conds):
    sndfiles[timbre_cond] = {}
    for sync_cond in sync_conds:
        sndfiles[timbre_cond][sync_cond] = {}
        sndfiles[timbre_cond][sync_cond] = [elem for elem in all_snds[i] if os.path.basename(elem).startswith(sync_cond)] 
# now to access all no_timbre sndfiles, e.g. sndfiles_no = sndfiles['no']['none'] + sndfiles['no']['weak'] + ... 
# or like this:
# sndfiles_no_timbre = []
# for sync_cond in sync_conds:
#   sndfiles_no_timbre.extend(sndfiles['no'][sync_cond])

# A1, A2 have all the audio 
versions = ['psychopy-A1/', 'psychopy-A2/']
blocks = ['block_1', 'block_2']
allstim = []
for version in versions:
    for block in blocks:
        for audiofi in glob.glob(os.path.join('psychopy/swarm-tapping-study', version, block, '*.wav')):
            audiofilename = str.split(os.path.basename(audiofi), '.')[0]
            print(audiofilename)
            allstim.append(audiofilename)

allnotimbre = [elem for elem in allstim if str.split(elem,'_')[1] == 'n'] 
alltimbre = [elem for elem in allstim if str.split(elem,'_')[1] == 't']

n_strong = [elem for elem in allnotimbre if str.split(elem,'_')[0] == 'strong']
n_medium = [elem for elem in allnotimbre if str.split(elem,'_')[0] == 'medium']
n_weak = [elem for elem in allnotimbre if str.split(elem,'_')[0] == 'weak']
n_none = [elem for elem in allnotimbre if str.split(elem,'_')[0] == 'none']

t_strong = [elem for elem in alltimbre if str.split(elem,'_')[0] == 'strong']
t_medium = [elem for elem in alltimbre if str.split(elem,'_')[0] == 'medium']
t_weak = [elem for elem in alltimbre if str.split(elem,'_')[0] == 'weak']
t_none = [elem for elem in alltimbre if str.split(elem,'_')[0] == 'none']



#%% ################## get center periods from data txt files from generative model 

sync_cond = ['none', 'weak', 'medium', 'strong']
#syncbatch = [sndfiles['no']['none'], sndfiles['no']['weak'], sndfiles['no']['medium'], sndfiles['no']['strong']]

idealperiods = {}
sndbeatbins = {}

# load beatbins for no-timbre type
datadirs = ['stim-no-timbre-5', 'stim-timbre-5']
timbre_tags = ['n','t']
stimuli_dirs = ['stimuli_1', 'stimuli_2', 'stimuli_3', 'stimuli_4']
#beatbins_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows')
#centerbpms_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm')

sndbeatbins = {}
centerbpms = {}
centerperiods = {}

for datadir, ttag in zip(datadirs, timbre_tags):
    for stimuli_dir in stimuli_dirs:
        beatbins_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows')
        for fi in glob.glob(beatbins_dir + '/*.txt'):
            fi_basename = str.split(os.path.basename(fi), '.')[0] # --> weak_79_1
            f = str.split(fi_basename, '_') 
            sync_cond = "_".join([f[0], ttag, f[1], f[2]])
            thebeatbins = np.loadtxt(fi, delimiter='\n')
            sndbeatbins[sync_cond] = thebeatbins
        
        centerbpms_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm')    
        for fi in glob.glob(centerbpms_dir + '/*.txt'):       
            fi_basename = str.split(os.path.basename(fi), '.')[0] # --> weak_79_1
            f = str.split(fi_basename, '_') 
            sync_cond = "_".join([f[0], ttag, f[1], f[2]])    
            thecenterbpm = np.loadtxt(fi)        
            centerbpms[sync_cond] = float(thecenterbpm)
            centerperiods[sync_cond] = 60./float(thecenterbpm)
            

#%% ######### parse subject taps in csv output files and format into dataframes or arrays

# default dictionarya
subject_resps = defaultdict(lambda: defaultdict(list))

ordered_subjects = []

# string prompts in header of csv files 
block1taps = 'block1_taps.rt'
block2taps = 'block2_taps.rt'
csv_sndfiles = 'sndfile'
csv_tempo = 'tempo'
csv_coupling_cond = 'cond'
csv_version = 'version'
csv_participant = 'Participant Initials'
csv_type = 'type'

##################################################
######### only take good USABLE csv files #####
#################################################
batch_folder = 'usable-batch-10-26'
#batch_folder = 'usable-stanford-batch'
#batch_folder = 'usable-mturk-batch'
subject = []
csvfiles = []   

for csv_ in glob.glob('psychopy/swarm-tapping-study/mturk-csv/' + batch_folder + '/*.csv'):
    namestripped = os.path.basename(csv_).split('.')[0].split(' ')[0]
    subject.append(namestripped)
    csvfiles.append(csv_)

#%%########## READ IN THE SUBJECT TAPS #################

for csv_file, person in zip(csvfiles, subject):
    print('SUBJECT: ', person)
    df_block = pd.read_csv(csv_file, keep_default_na=False)
    subject_resps[person] = {}  

    try:

        df_block_1 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block1taps])[4:44]
        df_block_2 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block2taps])[44:-1]
        
        df_block_1_type = df_block_1[csv_type]
        #timbre_type = df_block_1['sndfile'].values
    
        for index, row in df_block_1.iterrows():
            sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = []
        for index, row in df_block_2.iterrows():
            sync_cond_version  = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = []
    
        for index, row in df_block_1.iterrows():
            sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = removeStrFormatting(row[block1taps])
        for index, row in df_block_2.iterrows():
            sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
            subject_resps[person][sync_cond_version] = removeStrFormatting(row[block2taps])
    
    except TypeError:
        print('could not read %r csv file' %(person))
        
       
#####NB: subject_resps are now in this format 
#### subject_resps[person][type(no, timbre)][sync_tempo_version]            
        

#%% ########## GET ALL STIM NAMES with full path from allstims dir --> allstims list
allstims = []
for fi in glob.glob('./psychopy/swarm-tapping-study/allstims/*.wav'):
    allstims.append(fi)
## allstims is full file path of every stimuli 

#%% ### reformat trials subjects did not perform with empty list '' -> [] ###
  
subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/' + batch_folder + "/subjects/"

# replace all empty trials with [] (tried with np.nan but not good for plotting... )
for person in subject:
    print(person)
    for n, sndfile in enumerate(allstims):
        sync_cond_version = str.split(os.path.basename(sndfile), '.')[0]
    
        try:
            if (subject_resps[person][sync_cond_version] == ['']):
                subject_resps[person][sync_cond_version] = []
        except KeyError:
            print('subject %r did not tap to %r' %(person, sndfile))

#%%% ########### UTIL FUNCTIONS FOR BEAT BINNING ########
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

#%% other way of binning beats when multiple taps in same bin 
# def binBeats(taps, beat_bins):
#     taps = np.array(taps)
#     digitized = np.digitize(taps, beat_bins) # in secs, returns which beat bin each tap should go into
#     bins = [taps[digitized == i] for i in range(1, len(beat_bins)+1)]
#     return bins

# def binTapsFromBeatWindow(taps):
#     binnedtaps = []
#     avg_taps_per_bin = []
#     for i, tap in enumerate(taps):
#         try:
#             num_taps_in_bin = len(taps[i])
#             avg_taps_per_bin.append(num_taps_in_bin)

#             if num_taps_in_bin > 1:   
#                 random_tap = np.random.randint(low=0, high=num_taps_in_bin)
#                 binnedtaps.append(taps[i][random_tap]) # take random tap in window
#             if num_taps_in_bin == 0:
#                 binnedtaps.append(np.nan)
#             if num_taps_in_bin == 1:
#                 binnedtaps.append(taps[i][0])
#         except IndexError:
#             binnedtaps.append(np.nan)
    
#     avg_taps_per_stim = np.mean(avg_taps_per_bin)
#     return binnedtaps, avg_taps_per_stim
#%% ############################################################
##################### ITI of beat sections ########################
##############################################################


sndfile_conds = ['none','weak','medium','strong']   # strings for coupling cond
timbre_conds = [t_none, t_weak, t_medium, t_strong]
notimbre_conds = [n_none, n_weak, n_medium, n_strong]
############ IMPORTANT ###############################################
############## NOW DOING WHOLE BATCH (TIMBRE AND NON TIMBRE) AT ONCE #######
#####################################################################
all_timbre_conds = [timbre_conds, notimbre_conds]
t_strs = ['t', 'n']
###############################################################


#%% ####################################################
############ ITI ANALYSIS ##############################
########################################################

# make directories for individual ITI subject plots
subject_iti_dir = './psychopy/swarm-tapping-study/analysis-scripts/plots/usable-batch/subject-10-6-plots/'
for person in subject:
    the_subject_iti_dir = subject_iti_dir + person
    if os.path.exists(the_subject_iti_dir) == False:
        print('making dir for ', person, ' in ', the_subject_iti_dir)
        os.mkdir(the_subject_iti_dir)  

# this function plots each subjects individual ITIs per beat and saves into their subject directory  
def plotSubjectITIperTap(means, stds, label_str, person_str):
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    ax.plot(means)
    xrange = np.linspace(0, len(means) -1 , len(means))
    ax.errorbar(xrange, means, yerr=stds, marker='.', label=label_str, capsize=3)
    ax.set_title(' '.join([person_str, sync_str]))
    plt.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/usable-batch/subject-10-6-plots/' + person_str + '/' + person_str + ' ' + label_str + '.png', dpi=160)

#%% get all the trigger points in generative model to run ITIs on it 
# phases = []
# np.load('./all-stim-all-zcs/' + sync_cond_version + '.npy', allow_pickle=True)

#%%INTIALIZE DICTIONARIES BEFORE LOOP 
suptitle_str = 'Timbre and No Timbre Condition' # e.g. 'Timbre Condition', "No Timbre Condition"
binned_subject_taps = {}
subject_sync_cond_taps, subject_sync_cond_itis = {}, {}

for person in subject:
    subject_sync_cond_taps[person], subject_sync_cond_itis[person] = {}, {}

    for sync_str in sync_conds:
        subject_sync_cond_taps[person][sync_str], subject_sync_cond_itis[person][sync_str] = {}, {}
        for version, v in zip(all_timbre_conds, t_strs):
            subject_sync_cond_taps[person][sync_str][v] = []
            subject_sync_cond_itis[person][sync_str][v] = []

#%% ######## GET ITI OF STIMULI FROM GENERATIVE MODEL ##########

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
            y, _ = librosa.load('./psychopy/swarm-tapping-study/allstims/' + sync_cond_version + '.wav')
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
plt.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/gen-model/iti-gen-model.png', dpi=130)
#%%
### NB: weak and medium don't appear to be that different in terms of average ITI per beat section  
### main difference is their sx sizes 
#### look below at difference between weak and medium 
print('mean across beat sections of SD:')
for v in t_strs:
    print('for ', v, ' weak:', np.mean(iti_segment_sx['weak'][v]), 'medium:', np.mean(iti_segment_sx['medium'][v]))
              
#%%  plot ITIs of all oscillators from gen model conds for t and n on same plot  
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

#%%INTIALIZE DICTIONARIES BEFORE LOOP 
suptitle_str = 'Timbre and No Timbre Condition' # e.g. 'Timbre Condition', "No Timbre Condition"
def initializeDicts():
    binned_subject_taps = {}
    subject_sync_cond_taps, subject_sync_cond_itis = {}, {}
    
    for person in subject:
        subject_sync_cond_taps[person], subject_sync_cond_itis[person] = {}, {}
    
        for sync_str in sync_conds:
            subject_sync_cond_taps[person][sync_str], subject_sync_cond_itis[person][sync_str] = {}, {}
            for version, v in zip(all_timbre_conds, t_strs):
                subject_sync_cond_taps[person][sync_str][v] = []
                subject_sync_cond_itis[person][sync_str][v] = []
 

#%%NB: ONE THIS CODE BLOCK OR THE NEXt
##  BEAT BINNING:  GET ITIs FROM SUBJECT_RESPS[] WITH BEAT BINNING 
beat_binning_flag = 'w beat binning'       
for version, v in zip(all_timbre_conds, t_strs):
    for sync_cond, sync_str in zip(version, sync_conds):
        #subject_sync_cond_taps[person][sync_str][v] = [] 
        #subject_sync_cond_itis[person][sync_str][v] = []
        
        for sync_cond_version in sync_cond:
                        
            sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
            timbre_version = sync_cond_version.split('_')[1] # get the timbre tag
                        
            binned_subject_taps[sync_cond_version] = []
            
            for person in subject:
                try:
                    tap_resps = subject_resps[person][sync_cond_version] # get subject taps per stim 
                    binned_taps = binBeats(tap_resps, sndbeatbin)
                    # beat binning for ITI or no? 
                    binned_taps, _ = binTapsFromBeatWindow(binned_taps)
                    binned_subject_taps[sync_cond_version].append(binned_taps) # save subject's binned_taps per stim
                    # accumulate subject taps per sync_cond
                    subject_sync_cond_taps[person][sync_str][v].extend(binned_subject_taps[sync_cond_version])                    
                    # get normalized ITI vector and add it to the subject array
                    normalized_tap_iti = list(np.diff(binned_taps)/centerperiods[sync_cond_version])
                    #print(normalized_tap_iti)
                    subject_sync_cond_itis[person][sync_str][v].append(normalized_tap_iti)
                    #print(subject_sync_cond_itis[person][sync_str][v])

                except KeyError:
                    #print('did not tap to ', sync_cond_version)
                    pass
        
#%% NB: NO BEAT BINNING.......
beat_binning_flag = 'w NO beat binning'
for version, v in zip(all_timbre_conds, t_strs):
    for sync_cond, sync_str in zip(version, sync_conds):
        iti_segment_mx[sync_str], iti_segment_sx[sync_str] = {}, {}

        
        for sync_cond_version in sync_cond:
          
            sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
            timbre_version = sync_cond_version.split('_')[1] # get the timbre tag
                        
            binned_subject_taps[sync_cond_version] = []
            
            for person in subject:
                try:
                    tap_resps = subject_resps[person][sync_cond_version] # get subject taps per stim 
                    #binned_taps = binBeats(tap_resps, sndbeatbin)
                    # beat binning for ITI or no? 
                    #binned_taps, _ = binTapsFromBeatWindow(binned_taps)
                    #binned_subject_taps[sync_cond_version].append(binned_taps) # save subject's binned_taps per stim

                    # accumulate subject taps per sync_cond
                    subject_sync_cond_taps[person][sync_str][v].extend(tap_resps)
                    
                    # get normalized ITI vector and add it to the subject array
                    normalized_tap_iti = list(np.diff(tap_resps)/centerperiods[sync_cond_version])
                    #print(normalized_tap_iti)
                    subject_sync_cond_itis[person][sync_str][v].append(normalized_tap_iti)
                    #print(subject_sync_cond_itis[person][sync_str][v])

                except KeyError:
                    #print('did not tap to ', sync_cond_version)
                    pass

#%% NB: OUTLIER ALGORITHM 
beat_binning_flag = 'w algo outlier'
for version, v in zip(all_timbre_conds, t_strs):
    for sync_cond, sync_str in zip(version, sync_conds):
        iti_segment_mx[sync_str], iti_segment_sx[sync_str] = {}, {}

        
        for sync_cond_version in sync_cond:
          
            sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
            timbre_version = sync_cond_version.split('_')[1] # get the timbre tag
                        
            binned_subject_taps[sync_cond_version] = []
            
            for person in subject:
                try:
                    tap_resps = subject_resps[person][sync_cond_version] # get subject taps per stim 
                    tap_iti = np.diff(tap_resps)
                    tap_mean = np.nanmean(tap_iti)
                    tap_std = np.nanstd(tap_iti)
                    tap_resps_algo = [tap for tap in tap_iti if (tap < tap_mean + 2*tap_std) and (tap >= tap_mean - 2*tap_std)]
                    tap_resps_secs = [tap_resps[t] for t, tap in enumerate(tap_iti) if (tap < tap_mean + 2*tap_std) and (tap >= tap_mean - 2*tap_std)]
                    # mask = [True]
                    # tap_arr = []
                    # for z, tap in enumerate(tap_iti):
                    #     if (tap < tap_mean + 2*tap_std) and (tap >= tap_mean - 2*tap_std):
                    #         tap_arr.append(tap)
                    #         mask.append(True)
                    #     else:
                    #         mask.append(False)
                    #         pass
                    
                    #binned_taps = binBeats(tap_resps, sndbeatbin)
                    # beat binning for ITI or no? 
                    #binned_taps, _ = binTapsFromBeatWindow(binned_taps)
                    #binned_subject_taps[sync_cond_version].append(binned_taps) # save subject's binned_taps per stim
                    
                    # accumulate subject taps per sync_cond
                    #mask.insert(0, True)
                    subject_sync_cond_taps[person][sync_str][v].extend(tap_resps_secs)
                    
                    # get normalized ITI vector and add it to the subject array
                    normalized_tap_iti = list(np.array(tap_resps_algo)/centerperiods[sync_cond_version])
                    #print(normalized_tap_iti)
                    subject_sync_cond_itis[person][sync_str][v].append(normalized_tap_iti)
                    #print(subject_sync_cond_itis[person][sync_str][v])

                except KeyError:
                    #print('did not tap to ', sync_cond_version)
                    pass
                    
                    
#%% #%% ############## NEW ITI PLOTS!!!! ##################
######################################################


####### ITI SLICING AND TAKE MEAN/STD

# how many beat sections to analyze?
#beatsegments = [(0,5), (5,10), (10,15), (15,20)] # 4, 5 beat sections
#beatsegments = [(0,4), (4,8), (8,12), (12,16), (16,20)] # 5, 4 beat sections 
beatsegments = [(0,3), (3,6), (6,9), (9,12), (12, 15), (15,18)] # 6,3 beat sections
#beatsegments = [(0,2), (2,4), (4,6), (6,8), (8, 10), (10,12), (12,14), (14,16), (16, 18)] # 9, 2 beat sections


# dictionaries to hold mx, sx, and mx/sx errors 
iti_segment_mx, iti_segment_sx= {},{}
iti_mx, iti_sx = {}, {}
iti_mx_error, iti_sx_error = {}, {}

# beat str array for csv file output 
beat_strs = [str(i) for i in range(len(beatsegments))]

beat_segment_dir = './psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/' + str(len(beatsegments)) + '-beat-segments/' 
itis_dir = beat_segment_dir + '/ITIs/'
pcs_dir = beat_segment_dir + '/PCs/'
csvs_dir = beat_segment_dir + '/csvs/'

########## make all the directories #################
makeDir(beat_segment_dir)
makeDir(itis_dir) # make ITI subdir
makeDir(pcs_dir) # make PCs subdir 
makeDir(csvs_dir) # make csvs subdir 

# for PC directories, model and subjects subdirs and phases dir 
model_dir = pcs_dir + 'model/'
subject_dir = pcs_dir + 'subject/'
makeDir(model_dir)
makeDir(model_dir + 'phases/')
makeDir(subject_dir)
makeDir(subject_dir + 'phases/')

#####################################
# timestamp for csv file and cross referencing plot with csv 
now = datetime.datetime.now()
timestamp = str(now.month) + '-' + str(now.day) + '_' + str(now.hour) + '-' + str(now.minute)   

makeDir('./psychopy/swarm-tapping-study/analysis-scripts/ITIs/')

with open(csvs_dir + timestamp + '-' + beat_binning_flag + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["subject", "study", "version", "condition", "section", "mx", "sx"])
    
    for sync_str in sync_conds:
        iti_segment_mx[sync_str], iti_segment_sx[sync_str] = {}, {}
        iti_mx[sync_str], iti_sx[sync_str] = {}, {}
        iti_mx_error[sync_str], iti_sx_error[sync_str] = {}, {}
        
        for v in t_strs:
            iti_mx[sync_str][v], iti_sx[sync_str][v] = [], []
            iti_mx_error[sync_str][v], iti_sx_error[sync_str][v] = [], []
                        
            for person in subject:                
                study_number = person.split('_')[1].split('-')[1]

                df_taps = pd.DataFrame(subject_sync_cond_itis[person][sync_str][v])
                
                iti_segment_mx[sync_str][v], iti_segment_sx[sync_str][v] = [], []
                
                for beatseg, beat_str in zip(beatsegments, beat_strs):
                    tap_col = df_taps.iloc[:,beatseg[0]:beatseg[1]].values
                    # get mean, std
                    mx = np.nanmean(tap_col)
                    sx = np.nanmean(np.nanstd(tap_col, axis=1))
                    #sx = np.nanstd(tap_col)
                    #sx = np.nanstd(np.nanstd(tap_col, axis=1))
               
                    iti_segment_mx[sync_str][v].append(mx) 
                    iti_segment_sx[sync_str][v].append(sx)
                    
                    writer.writerow([person, study_number, v, sync_str, beat_str, mx, sx])
                
                # accumulate mean iti, and sd iti per person per sync_cond, nb: have to take means later
                iti_mx[sync_str][v].append(iti_segment_mx[sync_str][v])
                iti_sx[sync_str][v].append(iti_segment_sx[sync_str][v])

            subject_iti_means = np.array(iti_mx[sync_str][v])
            subject_iti_stds = np.array(iti_sx[sync_str][v])
                   
            # COMPUTER SUBJECT-TO-SUBJECT ERRORS FOR MX,SX PER SYNC_COND
            iti_mx_error[sync_str][v] = np.nanstd(subject_iti_means, axis=0)
            iti_sx_error[sync_str][v] = np.nanstd(subject_iti_stds, axis=0)
            
           
sns.set()
sns.set_palette(sns.color_palette("Paired"))

plt.figure()

fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(10,5), sharex=True, gridspec_kw={"height_ratios":[0.02,1,1]})

# labels for beat sections 
xlabels_pos = np.arange(0,len(beatsegments))
beatsectionlabels = [str(elem[0]) + '-' + str(elem[1]) for elem in beatsegments] # form str array for plotting

import matplotlib.ticker as ticker


for i, v in enumerate(t_strs):
    for j, sync_str in enumerate(sndfile_conds):
        iti_mx_arr = np.array(iti_mx[sync_str][v])
        iti_sx_arr = np.array(iti_sx[sync_str][v])
        
        subject_iti_means = np.nanmean(iti_mx[sync_str][v], axis=0)
        subject_iti_stds = np.nanmean(iti_sx[sync_str][v], axis=0)
        
        xrange = np.linspace(0, len(subject_iti_means) - 1, len(subject_iti_means))       
        # MEANS
        ax[1,i].plot(subject_iti_means, linewidth=0.8)
        ax[1,i].errorbar(xrange, subject_iti_means, yerr=iti_mx_error[sync_str][v], label=sync_str, marker='.', capsize=3)        
        # STDS
        ax[2,i].plot(subject_iti_stds, linewidth=0.8)
        ax[2,i].errorbar(xrange, subject_iti_stds, yerr=iti_sx_error[sync_str][v], label=sync_str, marker='.', capsize=3)        
        
        ax[2,i].set_title('ITI SD')
        ax[1,i].set_title('ITI MX')
        
        ### FORMATTING ###
        # ax[1,i].set_ylim([0.85, 2.0])
        # ax[2,i].set_ylim([-0.2, 1.5])

        ax[1,i].set_ylim([0.3, 1.5])
        ax[2,i].set_ylim([-0.1, 0.75])        
    
        ax[1,i].set_xticks(xlabels_pos)
        ax[2,i].set_xticklabels(beatsectionlabels, fontsize=6)
        ax[2,i].set_xlabel('beat segment')
        
        # turn on LOG SCALE? 
        #ax[1,i].set_yscale("log")
        #ax[2,i].set_yscale("log")
        
        # ax[1,i].set_xticklabels(str(round(float(label), 2)) , fontsize=6)
        # ax[2,i].set_xticklabels(beatsectionlabels, fontsize=6)
        #ax[1,i].ticklabel_format(style='sci', scilimits=(-6, 9))  # disable scientific notation
        #ax[1,i].set_major_formatter(ScalarFormatter())
        
        #ax[1,i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        #ax[2,i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        
        #ax[1,i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        #ax[2,i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))


# turn off scientific notation 
# import matplotlib.ticker as ticker
# for ax_ in ax[1:].flat: 
#     ax_.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
#     ax_.ticklabel_format(style='sci', scilimits=(-6, 9))  # disable scientific notation

cols = ['{}'.format(col) for col in ['Timbre', 'No Timbre']]
for ax, col in zip(ax[0], cols):
    ax.axis("off")
    ax.set_title(col, fontweight='bold')
    
plt.legend(title='coupling conditions', bbox_to_anchor=(1., 1.05))
fig.tight_layout()
plt.savefig(itis_dir + timestamp + ' timbre-no-timbre-ITI-SD-' + beat_binning_flag + '.png', dpi=160)


#%%
#########################################################################
################### PHASE COHERENCE ANALYSIS ###########################################
###############################################################################




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



         

sns.set()
sns.set_palette(sns.color_palette("Paired"))

#pc_beat_windows = [(0,4),(4,8),(8,12),(12,16)] # beat windows to form beat columns 


binned_taps_per_cond = {}
subject_binned_taps_per_cond = {}
subject_binned_taps_per_stim = {}

all_osc_binned_taps_per_stim = {}

all_subject_binned_taps_per_stim = {}
all_subject_binned_taps_per_cond = {}

all_subject_taps_per_cond = {}

plt.figure()

fig_subject, ax_subject = plt.subplots(nrows=8, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)
plt.figure()
fig_model, ax_model = plt.subplots(nrows=8, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)

plt.figure()
fig_combined, ax_combined = plt.subplots(nrows=8, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)


for ax_s, ax_m, ax_c in zip(ax_subject.flat, ax_model.flat, ax_combined.flat):
    ax_s.set_thetagrids([])
    ax_s.set_yticklabels([])
    ax_s.set_axisbelow(True)
    ax_s.grid(linewidth=0.1, alpha=1.0)

    ax_m.set_thetagrids([])
    ax_m.set_yticklabels([])
    ax_m.set_axisbelow(True)
    ax_m.grid(linewidth=0.1, alpha=1.0)

    ax_c.set_thetagrids([])
    ax_c.set_yticklabels([])
    ax_c.set_axisbelow(True)
    ax_c.grid(linewidth=0.1, alpha=1.0)



sns.set(style='darkgrid')

the_osc_phases = {}
osc_phases_cond = {}

random_color = np.random.random(4000)

R_csv = open('./psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/' + str(len(beatsegments)) + '-beat-segments/PCs/R_csv.csv', 'w')
R_writer = csv.writer(R_csv)
R_writer.writerow(['coupling condition', 'beat section', 'timbre version', 'R model', 'R subject', 'psi model', 'psi subject'])

vc = 0
for version,v in zip(all_timbre_conds, ['t','n']):
    the_osc_phases[v] = {}
    osc_phases_cond[v] = {}
    
    all_subject_binned_taps_per_stim[v], all_subject_binned_taps_per_cond[v] = {}, {}
    all_subject_taps_per_cond[v] = {}
    
    
    sc = 0
    for sync_cond, sync_str in zip(version, sync_conds):
        osc_phases_cond[v][sync_str] = []
        all_subject_binned_taps_per_cond[v][sync_str] = []
        
        for n, sync_cond_version in enumerate(sync_cond):
            print('working on %r %r/%r'%(sync_cond_version, n, len(sync_cond)))
            
            osc_phases = {}
            stim_phases_sec = {}
            
            sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])
            y, _ = librosa.load('./psychopy/swarm-tapping-study/allstims/' + sync_cond_version + '.wav')
            phases = np.load('./all-stim-all-zcs/' + sync_cond_version + '.npy', allow_pickle=True)

            the_osc_phases[v][sync_cond_version] = []

            ################## GENERATIVE MODEL ##################################
            for p, osc in enumerate(phases):
                binned_zcs = binBeats(osc, sndbeatbin)
                binned_zcs, _ = binTapsFromBeatWindow(binned_zcs)
                osc_phases[str(p)] = []
                
                for i in range(1, len(sndbeatbin)):
                    zctobin = binned_zcs[i-1]
                    binmin = sndbeatbin[i-1]
                    binmax = sndbeatbin[i]
                    bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                    osc_phases[str(p)].append(float(bininterp(zctobin)))
                
                the_osc_phases[v][sync_cond_version].append(osc_phases[str(p)])
            
            osc_phases_cond[v][sync_str].extend(the_osc_phases[v][sync_cond_version])
            
            ################# SUBJECTS TAPS ###################################
            all_subject_binned_taps_per_stim[v][sync_cond_version] = []  
            
            for person in subject:
                try:
                    taps = subject_resps[person][sync_cond_version]
                    binned_taps = binBeats(taps, sndbeatbin)
                    binned_taps, avg_taps_per_bin = binTapsFromBeatWindow(binned_taps) 
                    subject_binned_taps_per_stim[person] = []
                    
                    for i in range(1, len(sndbeatbin)):
                        taptobin = binned_taps[i-1]
                        binmin = sndbeatbin[i-1]
                        binmax = sndbeatbin[i]
                        bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                        subject_binned_taps_per_stim[person].append(float(bininterp(taptobin)))
                        
                    all_subject_binned_taps_per_stim[v][sync_cond_version].append(subject_binned_taps_per_stim[person])
      
                except:
                    pass
                                
            all_subject_binned_taps_per_cond[v][sync_str].extend(all_subject_binned_taps_per_stim[v][sync_cond_version])        
                
            ##################################################
        df_model  = pd.DataFrame(osc_phases_cond[v][sync_str]) 
        df_subject = pd.DataFrame(all_subject_binned_taps_per_cond[v][sync_str])                              
                     
        #### COLORS FOR SUBJECT AND MODEL ###########                       
        if v == "t":
            model_version_color = 'steelblue'
            subject_version_color = 'steelblue' 
            
        else:
            model_version_color = 'firebrick'
            subject_version_color = 'firebrick' 
            
            #subject_comb_version_color = 'mediumseagreen'
            
        for m, beatwindow in enumerate(beatsegments):
            # SUBJECT
            subject_beat_column = df_subject.iloc[:, beatwindow[0]:beatwindow[1]].values
            subject_beat_column_pooled_taps = subject_beat_column.flatten()
            
            # MODEL
            model_beat_column = df_model.iloc[:, beatwindow[0]:beatwindow[1]].values
            model_beat_column_pooled_taps = model_beat_column.flatten()            
            
            # calculate SUBJECT phase coherence
            R_subject = np.abs(np.nanmean(np.exp(1j*subject_beat_column_pooled_taps)))
            psi_subject = np.angle(np.nanmean(np.exp(1j*subject_beat_column_pooled_taps)))
            
            # calculate MODEL phase coherence
            R_model = np.abs(np.nanmean(np.exp(1j*model_beat_column_pooled_taps)))
            psi_model = np.angle(np.nanmean(np.exp(1j*model_beat_column_pooled_taps)))
            
            randomnoise_subject = np.random.random(len(subject_beat_column_pooled_taps))*0.3
            randomnoise_model = np.random.random(len(model_beat_column_pooled_taps))*0.3
    
            # PLOT AX OF SUBJECT
            ax_subject[sc + vc,m].scatter(subject_beat_column_pooled_taps, 0.7-randomnoise_subject, s=12, alpha=0.08, c=subject_version_color, marker='.', edgecolors='none')
            ax_subject[sc + vc,m].arrow(0, 0.0, psi_subject, R_subject, color='firebrick', linewidth=1)
 
            # PLOT AX OF MODEL 
            ax_model[sc + vc,m].scatter(model_beat_column_pooled_taps, 1-randomnoise_model, s=12, alpha=0.05, c=model_version_color, marker='.', edgecolors='none')
            ax_model[sc + vc,m].arrow(0, 0.0, psi_model, R_model, color='black', linewidth=1)
            
            # COMBINED SUBJECT + MODEL 
            ax_combined[sc + vc,m].scatter(subject_beat_column_pooled_taps, 0.7-randomnoise_subject, s=12, alpha=0.05, c='blueviolet', marker='.', edgecolors='none', zorder=0)
            ax_combined[sc + vc,m].arrow(0, 0.0, psi_subject, R_subject, color='red', linewidth=0.7, zorder=2)            
            ax_combined[sc + vc,m].scatter(model_beat_column_pooled_taps, 1-randomnoise_model, s=12, alpha=0.05, c='steelblue', marker='.', edgecolors='none', zorder=0)
            ax_combined[sc + vc,m].arrow(0, 0.0, psi_model, R_model, color='black', linewidth=0.7, zorder=1)

            ##### WRITE POOLED PHASES PER BEAT SEG TO TXT FILE TO RUN STATS IN R script 
            model_phases_txtfile = model_dir + "/phases/" + v + '-' + sync_str + "-" + str(m) + ".txt"
            np.savetxt(model_phases_txtfile, model_beat_column_pooled_taps, delimiter=',')            
            subject_phases_txtfile = subject_dir + "/phases/" + v + '-' + sync_str + "-" + str(m) + ".txt" 
            np.savetxt(subject_phases_txtfile, subject_beat_column_pooled_taps, delimiter=',')

            ### write R and ang per beat to csv
            if psi_model < 0:
                psi_model += 2*np.pi
            if psi_subject < 0: 
                psi_subject += 2*np.pi
            R_writer.writerow([sync_str, str(m), str(v), R_model, R_subject, str(np.degrees(psi_model)), str(np.degrees(psi_subject))])            
                
        sc += 1
    vc += 4

R_csv.close()

fig_combined.suptitle('Timbre and No Timbre Phase Coherence Per Beat Segment')
fig_subject.suptitle('Subject Phase Coherence Per Beat Segment')
fig_model.suptitle('Generative Model Phase Coherence Per Beat Segment')



colabels = [str(beatsegment) for beatsegment in beatsegments]

for ax, col in zip(ax_subject[0], colabels):
    ax.set_title(col, fontsize=10)
for ax, col in zip(ax_model[0], colabels):
    ax.set_title(col, fontsize=10)
for ax, col in zip(ax_combined[0], colabels):
    ax.set_title(col, fontsize=10)



rowlabels = ['none', 'weak', 'medium', 'strong', 'none', 'weak', 'medium', 'strong']
cnt = 0
for ax, row in zip(ax_subject[:,0], rowlabels):
    ax.set_ylabel(row, rotation=90, size='large', fontsize=8)
    
for ax, row in zip(ax_model[:,0], rowlabels):
    ax.set_ylabel(row, rotation=90, size='large', fontsize=8)
for ax, row in zip(ax_combined[:,0], rowlabels):
    ax.set_ylabel(row, rotation=90, size='large', fontsize=8)
   
fig_model.text(0.5, 0.04, 'beat segment', ha='center', va='center')
fig_subject.text(0.5, 0.04, 'beat segment', ha='center', va='center')
fig_combined.text(0.5, 0.04, 'beat segment', ha='center', va='center')
     
#plt.savefig(model_dir + 'gen model distributions.png', dpi=160)

fig_subject.savefig(subject_dir + 'subject  distributions.png', dpi=200)

fig_model.savefig(model_dir + 'model distributions.png', dpi=200)

fig_combined.savefig(model_dir + '../subject and model distributions.png', dpi=200)


#%%






