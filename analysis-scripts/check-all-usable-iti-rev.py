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

# def removeStrFormatting(str_array):
#     float_array = [] # array to hold new formatted lists    
#     str_array = [i for i in str_array if i] # remove empty elems
#     for str_arr in str_array:
#         str_arr = str_arr[1:-1] # remove "'[" and "]'"
#         str_arr = str.split(str_arr, ',') # split strings
#         str_arr = [float(elem) for elem in str_arr] # cast each str as float
#         #str_arr = np.array(str_arr, dtype=np.float32) # str to float
#         float_array.append(str_arr)
#     return float_array

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


#### only take good USABLE csv files #####
subject = []
csvfiles = []   
batch_folder = 'usable-batch-11-8'

for csv in glob.glob('psychopy/swarm-tapping-study/mturk-csv/' + batch_folder + '/*.csv'):
    namestripped = os.path.basename(csv).split('.')[0].split(' ')[0]
    subject.append(namestripped)
    csvfiles.append(csv)

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
                binnedtaps.append(taps[i][random_tap]) # take first tap in window
        except IndexError:
            binnedtaps.append(np.nan)
    
    avg_taps_per_stim = np.mean(avg_taps_per_bin)
    return binnedtaps, avg_taps_per_stim

#%%
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



#%%INTIALIZE DICTIONARIES BEFORE LOOP 
suptitle_str = 'Timbre and No Timbre Condition' # e.g. 'Timbre Condition', "No Timbre Condition"
binned_subject_taps = {}
subject_sync_cond_taps, subject_sync_cond_itis = {}, {}

for person in subject:
    subject_sync_cond_taps[person], subject_sync_cond_itis[person] = {}, {}

    for sync_str in sync_conds:
        subject_sync_cond_taps[person][sync_str], subject_sync_cond_itis[person][sync_str] = {}, {}
        for version, v in zip(all_timbre_conds, ['t', 'n']):
            subject_sync_cond_taps[person][sync_str][v] = []
            subject_sync_cond_itis[person][sync_str][v] = []

#%%  GET ITIs FROM SUBJECT_RESPS[]         
for version, v in zip(all_timbre_conds, ['t', 'n']):
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
                    binned_taps, _ = binTapsFromBeatWindow(binned_taps)
                    # save subject's binned_taps per stim
                    binned_subject_taps[sync_cond_version].append(binned_taps)
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
                    

#%% #%% ############## NEW ITI PLOTS!!!! ##################
######################################################

#%%
def makeDir(dirname):
    if os.path.exists(dirname) == False:
        print('making directory: ', dirname)
        os.mkdir(dirname)

#%%  ITI SLICING AND TAKE MEAN/STD

# how many beat sections to analyze?
#beatsegments = [(0,5), (5,10), (10,15), (15,20)] # 5 beat sections
beatsegments = [(0,4), (4,8), (8,12), (12,16), (16,20)] # 4 beat sections 
#beatsegments = [(0,3), (3,6), (6,9), (9,12), (12, 15), (15,18)] # 3 beat sections
#beatsegments = [(0,2), (2,4), (4,6), (6,8), (8, 10), (10,12), (12,14), (14,16), (16, 18)] # 2 beat sections


# dictionaries to hold mx, sx, and mx/sx errors 
iti_segment_mx, iti_segment_sx= {},{}
iti_mx, iti_sx = {}, {}
iti_mx_error, iti_sx_error = {}, {}

# beat str array for csv file output 
beat_strs = [str(i) for i in range(len(beatsegments))]

beat_segment_ITI_dir = './psychopy/swarm-tapping-study/analysis-scripts/plots/ITIs/' + str(len(beatsegments)) + '-beat-segments/' 
makeDir(beat_segment_ITI_dir)
#%%

# timestamp for csv file and cross referencing plot with csv 
now = datetime.datetime.now()
timestamp = str(now.month) + '-' + str(now.day) + '_' + str(now.hour) + '-' + str(now.minute)   

makeDir('./psychopy/swarm-tapping-study/analysis-scripts/ITIs/')

with open(beat_segment_ITI_dir + ' beat sections ' + timestamp + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["subject", "study", "version", "condition", "section", "mx", "sx"])
    
    for sync_str in sync_conds:
        iti_segment_mx[sync_str], iti_segment_sx[sync_str] = {}, {}
        iti_mx[sync_str], iti_sx[sync_str] = {}, {}
        iti_mx_error[sync_str], iti_sx_error[sync_str] = {}, {}
        
        for v in ['t','n']:
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
                    sx = np.nanstd(tap_col)
               
                    iti_segment_mx[sync_str][v].append(mx) 
                    iti_segment_sx[sync_str][v].append(sx)
                    
                    writer.writerow([person, study_number, v, sync_str, beat_str, mx, sx])
                
                iti_mx[sync_str][v].append(iti_segment_mx[sync_str][v])
                iti_sx[sync_str][v].append(iti_segment_sx[sync_str][v])

            subject_iti_means = np.array(iti_mx[sync_str][v])
            subject_iti_stds = np.array(iti_sx[sync_str][v])
            
            
            iti_mx_error[sync_str][v] = np.nanstd(subject_iti_means, axis=0)
            iti_sx_error[sync_str][v] = np.nanstd(subject_iti_stds, axis=0)
            
            

#%%
sns.set()
sns.set_palette(sns.color_palette("Paired"))
plt.figure()

fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(10,5), sharex=True, gridspec_kw={"height_ratios":[0.02,1,1]})

# labels for beat sections 
xlabels_pos = np.arange(0,len(beatsegments))
beatsectionlabels = [str(elem[0]) + '-' + str(elem[1]) for elem in beatsegments] # form str array for plotting


for i, v in enumerate(['t','n']):
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
        ax[1,i].set_ylim([0.5, 2.0])
        ax[2,i].set_ylim([-0.2, 1.5])
        
        #ax[1,i].set_ylim([0.8, 1.6])
        #ax[2,i].set_ylim([-0.2, 1.0])
        
        # ax[i,0].set_title('Timbre')
        # ax[i,1].set_title('No Timbre ')
        ax[1,i].set_xticks(xlabels_pos)
        ax[2,i].set_xticklabels(beatsectionlabels, fontsize=6)
        ax[2,i].set_xlabel('beat segment')

cols = ['{}'.format(col) for col in ['Timbre', 'No Timbre']]
for ax, col in zip(ax[0], cols):
    ax.axis("off")
    ax.set_title(col, fontweight='bold')
    
plt.legend(title='coupling conditions', bbox_to_anchor=(1., 1.05))
fig.tight_layout()
plt.savefig(beat_segment_ITI_dir + str(len(beat_strs)) + '-beats ' + timestamp + ' timbre-no-timbre-ITI-SD.png', dpi=160)
#%%



#%%######## NB: DON"T HAVE TO RUN, just checking participants tap info per beat window 
##################### how many taps on average make csv for avg beats per bin 
import csv
with open('./psychopy/swarm-tapping-study/mturk-csv/avg-beat-bins-both-cond.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["stimuli", "subject", "average tap in beat bins"])
    for person in subject:
        for sync_cond in sndfiles_batch:
            for sync_cond_version in sync_cond:
                try:
                    writer.writerow([sync_cond_version, person, avg_binned_taps[person][sync_cond_version]])
                except:
                    pass
#%% NB: DON"T HAVE TO RUN:::::: BEAT BIN AVG TOTALS CSV 
avg_tap_sync_cond = {}

with open('./psychopy/swarm-tapping-study/mturk-csv/avg-beat-bins-totals-subject.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["condition", "subject", "average tap in beat bins"])
    for person in subject:
        for sync_cond, sync_str in zip(sndfiles_batch, sndfile_conds):
            avg_tap_sync_cond[sync_str] = []
            for sync_cond_version in sync_cond:
                try:
                    avg_tap_sync_cond[sync_str].append(avg_binned_taps[person][sync_cond_version])
                    #writer.writerow([sync_cond_version, person, avg_binned_taps[person][sync_cond_version]])
                except:
                    pass
            avg_tap_sync = np.mean(avg_tap_sync_cond[sync_str])
            
            writer.writerow([sync_str, person, avg_tap_sync])




#%%

#%% #########################################################
################### PHASE COHERENCE ##########################
#############################################################


# for sync_cond, sync_str in zip(sndfiles_batch, sndfile_conds):
#     print(sync_str)
#     for sync_cond_version in sync_cond:
#         print('\t', sync_cond_version)
#         for person in subject:
#             print('\t \t', person)


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


#%% load up phases from .npy files from generative model 
stim_phases = {}
stim_R = {} 
stim_ang = {}

datadirs = ['./' + direct for direct in datadirs]

for datadir, ttag in zip(datadirs, timbre_tags):
    for stimuli_dir in stimuli_dirs:
        phases_dir = os.path.join(datadir, stimuli_dir, 'trigs', '*.npy')
        R_dir = os.path.join(datadir, stimuli_dir, 'phases', 'pc', '*.txt')
        ang_dir = os.path.join(datadir, stimuli_dir, 'phases', 'ang', '*.txt')
        for fi in glob.glob(phases_dir):
            print('working on importing phases from ', fi)
            filen = os.path.basename(fi).split('.')[0]
            filen_split = filen.split('_')
            filen_rev = filen_split[0] + '_' + ttag + '_' + filen_split[1] + '_' + filen_split[2]
            stim_phases[filen_rev] = np.load(fi, allow_pickle=True)
#%% NB: don't have to run, sanity check 
sync_cond_version = 'medium_n_119_1'
y, _ = librosa.load('./psychopy/swarm-tapping-study/allstims/' + sync_cond_version + '.wav')

trigs_p = np.load('./stim-no-timbre-5/stimuli_1/phases/medium_119_1.npy')

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(30,10))

#y_secs = librosa.samples_to_time(y[:22050*2])
zcs_arr = []
ax[0].plot(y[:22050*2], linewidth=0.7)
for osc in trigs_p[:10,:22050*2]:
    zcs = np.nonzero(osc)[0]
    #zcs = librosa.samples_to_time(zcs)
    ax[0].vlines(zcs, -1, 1, linewidth=0.3, color='green')
    zcs_arr.append(list(zcs))
    
trigs_new = np.load('./all-stim-all-zcs/' + sync_cond_version + '.npy', allow_pickle=True)
trigs_new = [[elem for elem in osc if elem <= 22050*2] for osc in trigs_new]
ax[1].plot(y[:22050*2], linewidth=0.7)

for osc in trigs_new[:10]:
    ax[1].vlines(osc, -1, 1, linewidth=0.3, color='green')



    #zcs_arr.append(list(zcs))

    
#%% ::::: DON'T RUN::::: just get trigger sample points from phases npy files and save to new directory with new _n_ _t_ naming convention 
stimuli_dir = ['stimuli_1/', 'stimuli_2', 'stimuli_3', 'stimuli_4']
source_dir = './all-stim-all-zcs/'
n_phases_dir = './stim-no-timbre-5/'
t_phases_dir = './stim-timbre-5/'

# for no timbre
for versiondir, tflag in zip([n_phases_dir, t_phases_dir], ["_n_", "_t_"]):

    for stimdir in stimuli_dir:
        for fi in glob.glob(versiondir + '/' + stimdir + '/phases/*.npy'):
            fn = os.path.basename(fi)
            newfilename = fn.split('_')[0] + tflag + fn.split('_')[1] + "_" + fn.split('_')[2]
            newfilename = source_dir + newfilename
    
            phases = np.load(fi)
            
            print('reading version %r %r, transforming into %r'%(tflag, fn, newfilename))
            
            zcs = []
            for osc in phases:
                zc = np.nonzero(osc)[0]
                zc = zc/22050.
                zcs.append(list(zc))        
            np.save(newfilename, zcs)
            
#%%
sns.set()
sns.set_palette(sns.color_palette("Paired"))
pc_beat_sections_dir = './psychopy/swarm-tapping-study/analysis-scripts/plots/PCs/' + str(len(beat_strs)) + '-beat-sections/'
makeDir(pc_beat_sections_dir)

model_dir = pc_beat_sections_dir + 'model/'
subject_dir = pc_beat_sections_dir + 'subject/'

makeDir(model_dir)
makeDir(subject_dir)
#%%
#pc_beat_windows = [(0,4),(4,8),(8,12),(12,16)] # beat windows to form beat columns 


binned_taps_per_cond = {}
subject_binned_taps_per_cond = {}
subject_binned_taps_per_stim = {}

all_osc_binned_taps_per_stim = {}

all_subject_binned_taps_per_stim = {}
all_subject_binned_taps_per_cond = {}

all_subject_taps_per_cond = {}

plt.figure()

fig, ax = plt.subplots(nrows=8, ncols=len(beatsegments), subplot_kw=dict(polar=True), gridspec_kw=
                            {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
                            figsize=(10,8), 
                            sharex=True)

for ax_ in ax.flat:
    ax_.set_thetagrids([])
    ax_.set_yticklabels([])
    ax_.set_axisbelow(True)
    ax_.grid(linewidth=0.1, alpha=1.0)

sns.set(style='darkgrid')

the_osc_phases = {}
osc_phases_cond = {}

random_color = np.random.random(4000)

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
            #stim_phases[sync_cond_version] = [[elem/22050. for elem in osc if elem <= len(y)] for osc in stim_phases[sync_cond_version]]
            #stim_phases[sync_cond_version] = librosa.samples_to_time(stim_phases[sync_cond_version])
            phases = np.load('./all-stim-all-zcs/' + sync_cond_version + '.npy', allow_pickle=True)

            the_osc_phases[v][sync_cond_version] = []

            ################## GENERATIVE MODEL #################
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
            
        #################### SUBJECTS TAPS ###################
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
                                            
        if v == "t":
            model_version_color = 'steelblue'
            subject_version_color = 'mediumseagreen'
        else:
            model_version_color = 'tomato'
            subject_version_color = 'mediumseagreen'
            
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
            ax[sc + vc,m].scatter(subject_beat_column_pooled_taps, 0.7-randomnoise_subject, s=12, alpha=0.05, c=subject_version_color, marker='.', edgecolors='none')
            ax[sc + vc,m].arrow(0, 0.0, psi_subject, R_subject, color='red', linewidth=1)
            #print(sync_str, m, R_model)
 
            # PLOT AX OF MODEL 
            ax[sc + vc,m].scatter(model_beat_column_pooled_taps, 1-randomnoise_model, s=12, alpha=0.05, c=model_version_color, marker='.', edgecolors='none')
            ax[sc + vc,m].arrow(0, 0.0, psi_model, R_model, color='black', linewidth=1)
            #print(sync_str, m, R_model)


        
        
        sc += 1
    vc += 4


colabels = [str(beatsegment) for beatsegment in beatsegments]
for ax_, col in zip(ax[0], colabels):
    ax_.set_title(col, fontsize=10)

rowlabels = ['none', 'weak', 'medium', 'strong', 'none', 'weak', 'medium', 'strong']
for ax_, row in zip(ax[:,0], rowlabels):
    ax_.set_ylabel(row, rotation=90, size='large', fontsize=8)
   
fig.text(0.5, 0.04, 'beat segment', ha='center', va='center')
     
#plt.savefig(model_dir + 'gen model distributions.png', dpi=160)
plt.savefig(model_dir + '../subject and model distributions.png', dpi=160)

#%%


c_ = 0
for sync_cond, sync_str in zip(all_timbre_conds, sndfile_conds):
    binned_taps_per_cond[sync_str] = [] 
    binned_taps_per_stim[sync_str] = {}
    
    subject_binned_taps_per_cond[sync_str] = []
       
    
    f = plt.figure()
    
    # NEED TO UNCOMMENT TO MAKE PLOTS PER STIMULUS
    # fig, ax = plt.subplots(len(sndfiles_batch[0]), 4, subplot_kw=dict(polar=True), gridspec_kw=
    #                         {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
    #                         figsize=(5,80), 
    #                         sharex=True)

    # fig, ax = plt.subplots(4, 4, subplot_kw=dict(polar=True), gridspec_kw=
    #                         {'wspace':0.2,'hspace':0.01,'top':0.9, 'bottom':0.1, 'left':0.125, 'right':0.9}, 
    #                         figsize=(5,4), 
    #                         sharex=True)

    plt.suptitle('all ' + sync_str + ' stimuli')
    # do these things to all axes in plot
    for ax_ in ax.flat:
        ax_.set_thetagrids([])
        ax_.set_yticklabels([])
        #ax_.set_rticks([])
        ax_.set_axisbelow(True)
        ax_.grid(linewidth=0.1, alpha=1.0)
    
    for n, sync_cond_version in enumerate(sync_cond): 
        print('working on ', sync_cond_version)
        
        y, _ = librosa.load('./psychopy/swarm-tapping-study/allstims/' + sync_cond_version + '.wav')
        numsamps = len(y)
        all_subject_binned_taps_per_stim[sync_cond_version] = []
        #binned_taps_per_stim[sync_str][sync_cond_version] = []        
        subject_binned_taps_per_stim = {}
        osc_binned_taps_per_stim = {}
        
        all_osc_binned_taps_per_stim[sync_cond_version] = []
        all_subject_binned_taps_per_stim[sync_cond_version] = []        
      
#        ax[n,0].set_ylabel(sync_cond_version, size=6, rotation=90)
                   
        sndbeatbin = librosa.samples_to_time(sndbeatbins[sync_cond_version])

               
        ################ GENERATIVE MODEL "taps" ##########
        for p, osc in enumerate(stim_phases[sync_cond_version][:, :numsamps]):
            zcs = np.nonzero(osc)[0]
            zcs = librosa.samples_to_time(zcs)
            binned_zcs = binBeats(zcs, sndbeatbin)
            binned_zcs, _ = binTapsFromBeatWindow(binned_zcs)
            osc_binned_taps_per_stim[str(p)] = []
            
            for i in range(1, len(sndbeatbin)):
                zctobin = binned_zcs[i-1]
                binmin = sndbeatbin[i-1]
                binmax = sndbeatbin[i]
                bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                osc_binned_taps_per_stim[str(p)].append(float(bininterp(zctobin)))
         
            all_osc_binned_taps_per_stim[sync_cond_version].append(osc_binned_taps_per_stim[str(p)])                  
        
        binned_taps_per_cond[sync_str].extend(all_osc_binned_taps_per_stim[sync_cond_version])
        
        ################ SUBJECT TAPS ######################
        for person in subject:
            try:
                taps = subject_resps[person][sync_cond_version]
                binned_taps = binBeats(taps, sndbeatbin)
                #print(sndbeatbin)
                #print(binned_taps)
                binned_taps, avg_taps_per_bin = binTapsFromBeatWindow(binned_taps) 
                #print(binned_taps)
                subject_binned_taps_per_stim[person] = []
                        
                for i in range(1, len(sndbeatbin)):
                    taptobin = binned_taps[i-1]
                    binmin = sndbeatbin[i-1]
                    binmax = sndbeatbin[i]
                    #print(str(binmin) + '\t' + str(binmax) + '\t taptobin: ' + str(taptobin))
                    bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) #map tap values within window from 0-2pi
                    subject_binned_taps_per_stim[person].append(float(bininterp(taptobin)))
                
                all_subject_binned_taps_per_stim[sync_cond_version].append(subject_binned_taps_per_stim[person])
                        
            except:
                #print('skipping ', person, sync_cond_version)
                pass

            
        subject_binned_taps_per_cond[sync_str].extend(all_subject_binned_taps_per_stim[sync_cond_version])
            
        
        df_subject = pd.DataFrame(all_subject_binned_taps_per_stim[sync_cond_version])
        df_model = pd.DataFrame(all_osc_binned_taps_per_stim[sync_cond_version])
        
        for m, beatwindow in enumerate(pc_beat_windows):
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
            
            randomnoise_subject = np.random.random(len(subject_beat_column_pooled_taps))*0.4
            randomnoise_model = np.random.random(len(model_beat_column_pooled_taps))*0.4

            
            #### PLOT SUBJECT PHASE COHERENCES
            # ax[n,m].scatter(subject_beat_column_pooled_taps, 2*np.pi+randomnoise_subject, s=12, alpha=0.75, c=random_color[:len(subject_beat_column_pooled_taps)] )
            # ax[n,m].arrow(0, 0.0, psi_subject, 2*np.pi*R_subject, color='black', linewidth=1)

            ##### PLOT MODEL PHASE COHERENCES            
            # ax[n,m].scatter(model_beat_column_pooled_taps, 2*np.pi+randomnoise_model, s=12, alpha=0.75, c=random_color[:len(model_beat_column_pooled_taps)] )
            # ax[n,m].arrow(0, 0.0, psi_model, 2*np.pi*R_model, color='black', linewidth=1)
            
            # if n==0:
            #     ax[n,m].set_title(str(beatwindow))
     
    ########################################################################################        
    ################### PLOT THE COMBINED STIMULI FOR COUPLING CONDITION ###################
    ########################################################################################
    df_model_cond = pd.DataFrame(binned_taps_per_cond[sync_str])
    df_subject_cond = pd.DataFrame(subject_binned_taps_per_cond[sync_str])
    
    
    for m, beatwindow in enumerate(pc_beat_windows):
        cond_beat_column = df_model_cond.iloc[:, beatwindow[0]:beatwindow[1]].values
        subject_beat_column = df_subject_cond.iloc[:, beatwindow[0]:beatwindow[1]].values
        #cond_beat_column_pooled_taps = np.nanmean(cond_beat_column, axis=0)
        cond_beat_column_pooled_taps = cond_beat_column.flatten() 
        subject_beat_column_pooled_taps = subject_beat_column.flatten()         
        
        R_cond = np.abs(np.nanmean(np.exp(1j*cond_beat_column_pooled_taps)))
        psi_cond = np.angle(np.nanmean(np.exp(1j*cond_beat_column_pooled_taps))) 
        randomnoise_cond = np.random.random(len(cond_beat_column_pooled_taps))*1.2 
        
        R_subject = np.abs(np.nanmean(np.exp(1j*subject_beat_column_pooled_taps)))
        psi_subject = np.angle(np.nanmean(np.exp(1j*subject_beat_column_pooled_taps))) 
        randomnoise_subject = np.random.random(len(subject_beat_column_pooled_taps))*1.2 
        
        # if timbre_dir == 'all':
        #                 #### write phases to txt file for R-script to run circular stats 
        #     model_phases_txtfile = "psychopy/swarm-tapping-study/analysis-scripts/R-scripts/csv/" + timbre_dir + "/all-" + sync_str + "-" + str(m) + ".txt"
        #     np.savetxt(model_phases_txtfile, cond_beat_column_pooled_taps, delimiter=',')
        # else:
        #### write phases to txt file for R-script to run circular stats 
        model_phases_txtfile = "psychopy/swarm-tapping-study/analysis-scripts/R-scripts/csv/model/" + timbre_dir + "/model-" + sync_str + "-" + str(m) + ".txt"
        np.savetxt(model_phases_txtfile, cond_beat_column_pooled_taps, delimiter=',')
        
        subject_phases_txtfile = "psychopy/swarm-tapping-study/analysis-scripts/R-scripts/csv/subject/" + timbre_dir + "/subject-" + sync_str + "-" + str(m) + ".txt"
        np.savetxt(subject_phases_txtfile, subject_beat_column_pooled_taps, delimiter=',') 
        
    


        ################ UNCOMMENT ONE OF THE FOLLOWING CODE BLOCKS BELOW #########################
       
        ##### PLOT MODEL PHASE COHERENCES  PER COUPLING COND              
        # ax[c_,m].scatter(cond_beat_column_pooled_taps, 
        #                   np.pi+randomnoise_cond, s=18, 
        #                   alpha=0.01, marker='.', c='royalblue', edgecolors='none') #, c=random_color[:len(model_beat_column_pooled_taps)] )
        # ax[c_,m].arrow(0, 0.0, psi_cond, np.pi*R_cond, color='royalblue', linewidth=1)       
        # ax[c_,0].set_ylabel(sync_str)
        # ax[0 ,m].set_title(str(m))

        ### PLOT SUBJECT PHASE COHERENCES PER COUPLING COND
        # ax[c_,m].scatter(subject_beat_column_pooled_taps, 
        #                   np.pi-randomnoise_subject, s=18, 
        #                   alpha=0.08, marker='.', c='crimson', edgecolors='none') 
        # ax[c_,m].arrow(0, 0.0, psi_subject, np.pi*R_subject, color='crimson', linewidth=1)       
        # ax[c_,0].set_ylabel(sync_str)
        # ax[0 ,m].set_title(str(m))
        

        #### PLOT SUBJECT and MODEL PHASE COHERENCES PER COUPLING COND ###############
        # ax[c_,m].scatter(subject_beat_column_pooled_taps, 
        #                   np.pi-randomnoise_subject, s=12, 
        #                   alpha=0.04, marker='v', c='crimson', edgecolors='none')

        # ax[c_,m].scatter(model_beat_column_pooled_taps, 
        #                   np.pi-randomnoise_model, s=12, 
        #                   alpha=0.05, marker='.', c='royalblue', edgecolors='none')
        
        # ax[c_,m].arrow(0, 0.0, psi_subject, np.pi*R_subject, color='crimson', linewidth=1)  
        # ax[c_,m].arrow(0, 0.0, psi_model, np.pi*R_model, color='royalblue', linewidth=1)       

        # ax[c_,0].set_ylabel(sync_str)
        # ax[0 ,m].set_title(str(m))
     
    c_ += 1

    
    #plt.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/PCs/' + timbre_dir +'/' + sync_str + '.png', dpi=160)
#plt.title('all ' + sync_str + ' beat segments')
#fig = plt.gcf()



# fig.savefig('./psychopy/swarm-tapping-study/analysis-scripts/plots/PCs/all-timbre-no-timbre/model-subject-all.png', dpi=160)
#plt.show()

#%%
fig, ax = plt.subplots(4,4, subplot_kw=dict(polar=True))
for m, beatwindow in enumerate(pc_beat_windows):
        model_beat_column = df_model.iloc[:, beatwindow[0]:beatwindow[1]].values
        model_beat_column_pooled_taps = model_beat_column.flatten() 
        R_model = np.abs(np.nanmean(np.exp(1j*model_beat_column_pooled_taps)))
        psi_model = np.angle(np.nanmean(np.exp(1j*model_beat_column_pooled_taps)))
        randomnoise_model = np.random.random(len(model_beat_column_pooled_taps))*0.4
        ax[0,m].scatter(model_beat_column_pooled_taps, 2*np.pi+randomnoise_model, s=12, alpha=0.75, c=random_color[:len(model_beat_column_pooled_taps)] )
        ax[0,m].arrow(0, 0.0, psi_model, 2*np.pi*R_model, color='black', linewidth=1)
#%%
fig, ax = plt.subplots(2,1, subplot_kw=dict(polar=True))

ax[0].scatter(np.array([0, np.pi/4]), np.array([np.pi/2,np.pi]), s=12)

#%% load up phases from .npy files from generative model 
stim_phases = {}
stim_R = {} 
stim_ang = {}

datadirs = ['./' + direct for direct in datadirs]

for datadir, ttag in zip(datadirs, timbre_tags):
    for stimuli_dir in stimuli_dirs:
        phases_dir = os.path.join(datadir, stimuli_dir, 'trigs', '*.npy')
        R_dir = os.path.join(datadir, stimuli_dir, 'phases', 'pc', '*.txt')
        ang_dir = os.path.join(datadir, stimuli_dir, 'phases', 'ang', '*.txt')
        for fi in glob.glob(phases_dir):
            print('working on importing phases from ', fi)
            filen = os.path.basename(fi).split('.')[0]
            filen_split = filen.split('_')
            filen_rev = filen_split[0] + '_' + ttag + '_' + filen_split[1] + '_' + filen_split[2]
            stim_phases[filen_rev] = np.load(fi)

        # for fi in glob.glob(R_dir):
        #     print('working on importing R from ', fi)
            
        #     filen = os.path.basename(fi).split('.')[0]
        #     filen_split = filen.split('_')
        #     filen_rev = filen_split[0] + '_' + ttag + '_' + filen_split[1] + '_' + filen_split[2]
        #     stim_R[filen_rev] = np.loadtxt(fi, delimiter=',')
        
        # for fi in glob.glob(ang_dir):
        #     print('working on importing ang from ', fi)
            
        #     filen = os.path.basename(fi).split('.')[0]
        #     filen_split = filen.split('_')
        #     filen_rev = filen_split[0] + '_' + ttag + '_' + filen_split[1] + '_' + filen_split[2]
        #     stim_ang[filen_rev] = np.loadtxt(fi, delimiter=',')

#%%

plt.plot(y)
plt.vlines(librosa.time_to_samples(sndbeatbin),-1,1, color='')

#%% sanity check, makesure phases and beat bins are properly formatted with audio file 
sndbeatbin = np.loadtxt('/Users/nolanlem/Documents/kura/kura-new-cond/py/stim-timbre-5/stimuli_4/phases/beat-windows/strong_119_4.txt')
phases = np.load('/Users/nolanlem/Documents/kura/kura-new-cond/py/stim-timbre-5/stimuli_4/phases/strong_119_4.npy')

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, 100)]


taps = []
plt.figure(figsize=(20,5))
for i,osc in enumerate(phases[:20]):
    nzidx = np.nonzero(osc)[0]
    #taps.append(nzidx)
    randcolor = np.random.randint(low=0, high=100)
    plt.vlines(nzidx[:20],-1,1, linewidth=1, color=colors[randcolor])
plt.vlines(sndbeatbin, -1,1, color='green')

y,sr_ = librosa.load('/Users/nolanlem/Documents/kura/kura-new-cond/py/stim-timbre-5/stimuli_4/strong_119_4.wav')
plt.plot(y)
plt.savefig('/Users/nolanlem/Desktop/wtf.png', dpi=160)

#%%         

allstims = []
for fi in glob.glob('./psychopy/swarm-tapping-study/allstims/*.wav'):
    allstims.append(fi)
    
    #%%
            
beatbins_filtered = {}
audio_sr = 22050.

for datadir, ttag in zip(datadirs, timbre_tags):
    for stimuli_dir in stimuli_dirs:
        phases_dir = os.path.join(datadir, stimuli_dir, 'phases', 'beat-windows', '*.npy')
        centerbpm_dir = os.path.join(datadir, stimuli_dir, 'phases', 'center-bpm', '*.npy')
        for fi in glob.glob(centerbpm_dir):
            tempo_cond = np.loadtxt(fi, delimiter=',')
            tempo_cond = audio_sr*1./(tempo_cond/60.)
            tempo_cond = 0.67*tempo_cond

            min_threshold_beatbin_size_ = int(0.67*audio_sr*60./tempo_cond_) # constrain bb to be < 0.5 of bpm center (in samples)
            min_threshold_beatbin_size = int(tempo_cond)
            
            beatbins = np.loadtxt()
    
for j,snd in enumerate(thebatch):
    print('working on ', snd)
    # v1 
    tempo_cond = snd.split(sep="_")[1]
    tempo_cond_ = float(tempo_cond)
    # v2 by center bpm  
    tempo_cond = np.loadtxt(audio_dir + "/phases/center-bpm/" + snd + ".txt", delimiter=',')        
    tempo_cond = audio_sr*1./(tempo_cond/60.)
    tempo_cond = 0.67*tempo_cond # can't be smaller than half the average bpm
    #print 'tempo cond_: %r \t tempo cond: %r'%(tempo_cond_, tempo_cond)
    
    min_threshold_beatbin_size_ = int(0.67*audio_sr*60./tempo_cond_) # constrain bb to be < 0.5 of bpm center (in samples)
    min_threshold_beatbin_size = int(tempo_cond)
    #print 'tempo cond_: %r \t tempo cond: %r'%(min_threshold_beatbin_size_, min_threshold_beatbin_size)

    beatbins = all_beat_bins[snd]
    beatbins_diffs = np.diff(beatbins)
    idx_to_delete = []
    for i, diff in enumerate(beatbins_diffs):
        if diff < (min_threshold_beatbin_size_):
            idx_to_delete.append(i)
    
    beatbins_filtered[snd] = np.delete(beatbins, idx_to_delete)
    
    y, sr_ = librosa.load(audio_dir + '/' + snd + ".wav", sr=int(audio_sr))
    ax[j].plot(y, linewidth=0.5)
    ax[j].vlines(beatbins_filtered[snd], -1, 1, color='red') 


               
            
                

                        
            
        # df_stim = pd.DataFrame(all_subject_binned_taps_per_stim[sync_cond_version])
        # #print('binned taps per stim',binned_taps_per_stim[sync_str][sync_cond_version])
        # # per subject ... 
        # subject_binned_taps_per_stim = [] 
        # for j in range(len(binned_taps_per_stim[sync_str][sync_cond_version])): # start analyzing beats on 1st window 
        #     # per beat column 
        #     for i in range(1,len(sndbeatbin)):
        #         taptobin = binned_taps_per_stim[sync_str][sync_cond_version][j][i-1]
        #         binmin = sndbeatbin[i-1]
        #         binmax = sndbeatbin[i]
        #         print(str(binmin) + '\t' + str(binmax) + '\t taptobin: ' + str(taptobin))
        #         bininterp = interp1d([binmin, binmax], [0, 2*np.pi]) # interpolate taps wrt beat bin to give value 0-2pi
        #         interpedtaps = bininterp(taptobin)
        #         print(interpedtaps)
        #         subject_binned_taps_per_stim.append(float(interpedtaps))
                
        #     all_subject_binned_taps_per_stim[sync_cond_version].append(subject_binned_taps_per_stim)
        
        # df_stim = pd.DataFrame(all_subject_binned_taps_per_stim[sync_cond_version])
                #print(interpedtaps)
            #circletaps.append(np.array(interpedtaps).mean())
            #alltaps.append(np.array(interpedtaps))    
            
        
        # df_stim = pd.DataFrame(binned_taps_per_stim[sync_str][sync_cond_version])
        
        # average_pc_per_stim = df_stim.mean(axis=0)
        
        # for beat_column in range(df_stim.shape[1]):
        #     df_stim.iloc[:, beat_column]
            
  #%%         
  
fig, ax = plt.subplots(2,1)
ax[0].scatter(np.array([1,2,3]), np.random.random(3))

            
            
            

#%%
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap
            
        
    
    
    
    









