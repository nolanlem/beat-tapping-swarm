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
#%%

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

#%%
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
#%%
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

       
#%% # get the csv files 

csvfiles = []
psychopy_batch = ['psychopy-A1','psychopy-A2','psychopy-B1','psychopy-B2']
mturk_batch1_all = 'psychopy/swarm-tapping-study/mturk-csv/batch1-all/'
versions = ['A1','A2','B1','B2']
mturk_batches_all = [os.path.join(mturk_batch1_all, version) for version in versions]

for version in versions:
    for file in glob.glob(os.path.join(mturk_batch1_all, version, "*.csv")):
        print(file)
        csvfiles.append(file)
#%%
# this is just to reformat the csv file that Takako gave me to pre-test the tap output
#csvfile = './psychopy/swarm-tapping-study/psychopy-A1/data/TFA1_kura-A1_2020-07-30_17h20.48.090.csv'

# csvfiles_ = []
# for fi in glob.glob('psychopy/swarm-tapping-study/practice-csvs/*.csv'):
#     csvfiles_.append(fi)

# for csvfile in csvfiles_:
#     csv_block = pd.read_csv(csvfile, keep_default_na=False)
#     thesndfiles = csv_block['sndfile'].values
#     thetype = csv_block['type']
    
#     newsndfilenames = []
#     for snd,t in zip(thesndfiles, thetype):
#         d = str.split(str.split(os.path.basename(snd), '.')[0],'_')
#         theblockpath = str.split(str.split(snd, '.')[0], '/')[0]
#         print('%r %r %r'%(theblockpath, d, t))
#         newpathname = os.path.join(theblockpath,'_'.join([d[0], t[0][0], d[1], d[2]]))
#         newfilename = newpathname + '.wav'
#         print(newfilename)
#         newsndfilenames.append(newfilename)
        
    
#     csv_block['sndfile'] = newsndfilenames
#     basename = os.path.basename(csvfile)
#     version = str.split(str.split(basename,'_')[1],'-')[1]
#     path2data = os.path.join('psychopy/swarm-tapping-study/psychopy-' + version, 'data', basename)
#     print(path2data)
#     csv_block.to_csv(path2data)

#%% get participant namesfrom csv

subject = []
for csv_file in csvfiles:
    csv_data = pd.read_csv(csv_file, keep_default_na=False)
    subject.append(csv_data['Participant Initials'][0])


  
#%%#### get subjects and csvfiles in lists, and make plots dirs for each subject 
subject = []
csvfiles = []
for exp_version in pp_stims:
    for datafi in glob.glob(os.path.join(exp_version,'data/*.csv')):
        csvfiles.append(datafi)
        theperson = str.split(os.path.basename(datafi), '_')[0]
        subject.append(theperson)
        subjectplot = './psychopy/swarm-tapping-study/analysis-scripts/plots/subjects/' + theperson # make dir for subject's individual plots
        if os.path.exists(subjectplot) == False:
            os.mkdir(subjectplot)
        
#%%
csv_data = pd.DataFrame(columns=['subject', 'condition', 'section', 'mx', 'sx', 'num obs'])


#####################################
# snfiles is full path to snd 
# all_snds is basename 
# dictionary of snds between no-timbre and timbre 

#%%
# timbre_conds = [sndfiles_no, sndfiles_timbre]

# for timbre_type in timbre_conds:
#     snds[timbre_type] = {}

#     no_none = [elem for elem in sndfiles_no if elem.startswith('none')]
#     no_weak = [elem for elem in sndfiles_no if elem.startswith('weak')]
#     no_medium = [elem for elem in sndfiles_no if elem.startswith('medium')]
#     no_strong = [elem for elem in sndfiles_no if elem.startswith('strong')]
#%% ################## FOR NO-TIMBRE TYPE

sync_cond = ['none', 'weak', 'medium', 'strong']
syncbatch = [sndfiles['no']['none'], sndfiles['no']['weak'], sndfiles['no']['medium'], sndfiles['no']['strong']]
beatsegments = [(0,5), (5,10), (10,15), (15,20), (20,25)]
beatstrings = ['0-5', '5-10', '10-15', '15-20', '20-25']

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
# old code
# for batch in syncbatch:
#     for snd in batch:
#         sndstr = os.path.basename(snd).split('.')[0]
#         wf, sr_ = librosa.load(snd)
#         beat_bins, avg_tempo, _ = formFixedBeatBins(wf, snd)
#         sndbeatbins[snd] = beat_bins
#         idealperiods[snd] = 60./avg_tempo # this is in BPM!!!   

#%% 


#%%
# parse subject taps in csv output files and format into dataframes or arrays

from collections import defaultdict

def removeStrFormatting(str_arr):
    str_arr = str_arr[1:-1] # remove "'[" and "]'"
    str_arr = str.split(str_arr, ',') # split strings
    try:
        str_arr = [float(elem) for elem in str_arr] # cast each str as float
    except ValueError:
        pass
    #str_arr = np.array(str_arr, dtype=np.float32) # str to float
    return str_arr

# default dictionarya

subject_resps = defaultdict(lambda: defaultdict(list))

ordered_subjects = []


block1taps = 'block1_taps.rt'
block2taps = 'block2_taps.rt'
csv_sndfiles = 'sndfile'
csv_tempo = 'tempo'
csv_coupling_cond = 'cond'
csv_version = 'version'
csv_participant = 'Participant Initials'
csv_type = 'type'

condition = []


for csv_file, person in zip(csvfiles, subject):
    print('SUBJECT: ', person)
    df_block = pd.read_csv(csv_file, keep_default_na=False)
    subject_resps[person] = {}  
    
    # get experiment version 
    # experiment_version = os.path.basename(csv_file).split('_')[1]
    # experiment_version = experiment_version.split('-')[1]
    # print(experiment_version)


    df_block_1 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block1taps])[4:44]
    df_block_2 = df_block.get([csv_participant, csv_sndfiles, csv_type, csv_coupling_cond, csv_tempo, csv_version, block2taps])[44:-1]
    
    #timbre_type = df_block_1['sndfile'].values

    for index, row in df_block_1.iterrows():
        sync_cond_version = str.split(os.path.basename(row[csv_sndfiles]), '.')[0]
        print(os.path.basename(row[csv_sndfiles]))
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

           
#####NB: subject_resps are now in this format 
#### subject_resps[person][type(no, timbre)][sync_tempo_version]            
        
#%% plot no-timbre taps in line with no-timbre audio 
# NB: this is only for NO-TIMBRE, have to redefine allblocks below 
# since TFA1 only did 'A1' version, we have to isolate the audio from that exp and 
# link it to the taps 

# this is ALL the stimuli for condition 
sndfiles_none_weak = sndfiles['no']['none'] + sndfiles['no']['weak'] 
sndfiles_med_strong = sndfiles['no']['medium'] + sndfiles['no']['strong']

allblocks = [sndfiles_none_weak, sndfiles_med_strong]

# A1: no_timbre (1,2)   timbre(1,2)


#%%

person = subject[0]

tap_snds_no = list(subject_resps[person].keys())

allstims = []
for fi in glob.glob('./psychopy/swarm-tapping-study/allstims/*.wav'):
    allstims.append(fi)
#%%#### beat plots per participant #####


#%%

## get all sounds full path for alltimbre and notimbre sound sets 
timbrepaths = []
notimbrepaths = []
for fi in alltimbre:
    timbrepaths.append(os.path.join('./psychopy/swarm-tapping-study/allstims/', fi + '.wav'))
for fi in allnotimbre :
    notimbrepaths.append(os.path.join('./psychopy/swarm-tapping-study/allstims/', fi + '.wav'))

#%%
    
A1_stimuli_names = []
A2_stimuli_names = []
B1_stimuli_names = []
B2_stimuli_names = []

for block in blocks:
    for fi in glob.glob('./psychopy/swarm-tapping-study/psychopy-A1/' + block + "/*.wav"):
        A1_stimuli_names.append(os.path.basename(fi))
    for fi in glob.glob('./psychopy/swarm-tapping-study/psychopy-A2/' + block + "/*.wav"):
        A2_stimuli_names.append(os.path.basename(fi))
    for fi in glob.glob('./psychopy/swarm-tapping-study/psychopy-B1/' + block + "/*.wav"):
        B1_stimuli_names.append(os.path.basename(fi))
    for fi in glob.glob('./psychopy/swarm-tapping-study/psychopy-B2/' + block + "/*.wav"):
        B2_stimuli_names.append(os.path.basename(fi))
#%%

thestim2plot = B1_stimuli_names
path2stims = './psychopy/swarm-tapping-study/allstims/'
plt.figure()
fig, ax = plt.subplots(len(thestim2plot), 1, sharex=True, sharey=True, figsize=(20,20))
person = 'cc' # change to plot specific participant 'bb' is A2 which is notimbre3,4 timbre3,4

    
subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/subjects/'

for n, sndfile in enumerate(B1_stimuli_names):

    sync_cond_version = str.split(os.path.basename(sndfile), '.')[0]
    y, sr_ = librosa.load(path2stims + sndfile)
    subjecttaps = librosa.time_to_samples(subject_resps[person][sync_cond_version]) #need to redefine [(no,timbre)] to look at timbre par exemple
    
    ax[n].plot(y, linewidth=0.5) # plot sound waveform
    ax[n].vlines(subjecttaps, -0.5, 0.5, color='red') # plot subject taps        
    ax[n].vlines(sndbeatbins[sync_cond_version], -1, 1, color='green') # plot beat bins 
    
    ax[n].set_yticks([]) # turn off y ticks

    fig.suptitle(person + ' Block ' + str(n+1) + ' taps ')


plt.savefig(os.path.join(subjectplotdir, person, '-t-block_ ' + str(n+1) + '-taps.png'), dpi=120)


#%%

path2stims = './psychopy/swarm-tapping-study/allstims/'
plt.figure()
fig, ax = plt.subplots(len(allstims), 1, sharex=True, sharey=True, figsize=(20,20))

for n, sndfile in enumerate(tap_snds_no):

    sync_cond_version = str.split(os.path.basename(sndfile), '.')[0]
    y, sr_ = librosa.load(path2stims + sndfile + '.wav')
    subjecttaps = librosa.time_to_samples(subject_resps[person][sync_cond_version]) #need to redefine [(no,timbre)] to look at timbre par exemple
    
    ax[n].plot(y, linewidth=0.5) # plot sound waveform
    ax[n].vlines(subjecttaps, -0.5, 0.5, color='red') # plot subject taps        
    ax[n].vlines(sndbeatbins[sync_cond_version], -1, 1, color='green') # plot beat bins 
    
    ax[n].set_yticks([]) # turn off y ticks


    fig.suptitle(person + ' Block ' + str(n+1) + ' taps ')
    plt.savefig(os.path.join(plotdir, person, '-t-block_ ' + str(n+1) + '-taps.png'), dpi=120)

#%% iti of beat sections 

subject_iti_slices = defaultdict(lambda: defaultdict(list))


# NB: need to load the average tempo and then normalize the iti to it 

for person in subject:
    subject_iti_slices[person] = {}
    for snd in sndfiles:
        sync_cond_version = str.split(os.path.basename(sndfile), '.')[0]
        subject_iti_slices[person][sync_cond_version] = []
        
        for beatseg in beatsegments:            
            taps = subject_resps[person][sync_cond_version]
            normalized_tap_iti = np.diff(taps)/centerperiods[sync_cond_version]
            tap_iti_slice = normalized_tap_iti[beatseg[0]:beatseg[1]]
            avg_tap_iti_slice = np.mean(tap_iti_slice) 
            
            subject_iti_slices[person][sync_cond_version].append(avg_tap_iti_slice) 

    
    
    
    
    









