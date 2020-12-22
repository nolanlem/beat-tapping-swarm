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
from shutil import copyfile
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
#%%
# A1 no(1,2)        timbre(1,2)
# B1 timbre(1,2)    no(1,2)
# A2 no(3,4)        timbre(3,4)
# B2 timbre(3,4)    no(3,4)
#

#%%

#%%

#%%
# A1, A2 have all the audio 
versions = ['psychopy-A1/', 'psychopy-A2/']
blocks = ['block_1', 'block_2']

       
#%% # get the csv files 

batch_folder = 'batch-12-7'###### batch where you've put the downloaded pavlovia experiment dirs
csvfiles = []


subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/' + batch_folder      


if os.path.exists(subjectplotdir) == False:
    os.mkdir(subjectplotdir)
    
subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/' + batch_folder + '/subjects/'     
if os.path.exists(subjectplotdir) == False:
    os.mkdir(subjectplotdir)

psychopy_batch = ['psychopy-A1','psychopy-A2','psychopy-B1','psychopy-B2']
renamed_batch = 'psychopy/swarm-tapping-study/mturk-csv/' + batch_folder + '-renamed/'
participants = []
versions = ['A1','A2','B1','B2']
#mturk_batches_all = [os.path.join(mturk_batch1_all, version) for version in versions]


for version in versions:
    for file in glob.glob('./psychopy/swarm-tapping-study/mturk-csv/' + batch_folder + '/' + version + "/*.csv"):
        print(file)
        csvfiles.append(file)
        participants.append(os.path.basename(file).split('.')[0])

#%% get participant namesfrom csv, rename according to if given 1. MTurk ID, 2. email, participant initials
mturk_prompts = ['If you are a MTurk worker, what is your MTurk worker ID?', "If you are a MTurk worker, what is your MTurk worker ID?'"]
email_prompt = "If you are not an MTurk worker, please provide your email address" 

subject = {}

email_subjects, mturk_subjects = [], []

for csv_file in csvfiles:
    csv_data = pd.read_csv(csv_file, keep_default_na=False)
    #subject.append(csv_data['Participant Initials'][0])
    
    csv_file_basename = os.path.basename(csv_file)
    
    mturk_id = 'none'
    email = 'none'
    
    try:
        mturk_id = csv_data[mturk_prompts[0]][0]
        #print(mturk_id)
    except:
        pass    
    try:
        mturk_id = csv_data[mturk_prompts[1]][0]
        #email = csvfile[email_prompt][0]            
        #print(mturk_id)
    except:
        pass

    try:
        email = csv_data[email_prompt][0]
    except:
        pass


    #print(mturk_id,email)    
    if (mturk_id != 'none') and (mturk_id != ''):
        mturk_subjects.append(csv_file)
        subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/' + batch_folder + '/subjects/' + mturk_id        
        subject[csv_file_basename] = mturk_id
        
        if os.path.exists(subjectplotdir) == False:
            os.mkdir(subjectplotdir)
    else:
        if (email != 'none') and (email != ''):
            print(email)
            email_subjects.append(csv_file)
            subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/' + batch_folder + '/subjects/' + email
            subject[csv_file_basename] = email           
            if os.path.exists(subjectplotdir) == False:
                os.mkdir(subjectplotdir)
        else:
            inits = csv_data['Participant Initials'][0] 
            print(inits)
            if (inits != 'none') and (inits != ''):
                subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/' + batch_folder + '/subjects/' + inits
                subject[csv_file_basename] = inits                                  
                if os.path.exists(subjectplotdir) == False:
                    os.mkdir(subjectplotdir)
            else:
                print(csv_file)
                subjectplotdir = './psychopy/swarm-tapping-study/analysis-scripts/plots/' + batch_folder + '/subjects/' + csv_file
                subject[csv_file_basename] = csv_file
 
                if os.path.exists(subjectplotdir) == False:
                    os.mkdir(subjectplotdir)
                   
        

#%% ################## GET BEAT BINS and CENTER BPM FROM GEN MODEL ##############

idealperiods, sndbeatbins = {},{}

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


#%% ####### GET SUBJECT TAPS in csv output files and format into dataframes or arrays

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

    try:

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
    
    except TypeError:
        print('could not read %r csv file' %(person))
        

        

           
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


#%% ##### GET ALL THE STIMS names from allstims dir --> allstims list
allstims = []
for fi in glob.glob('./psychopy/swarm-tapping-study/allstims/*.wav'):
    allstims.append(fi)
## allstims is full file path of every stimuli 
#%%#### beat plots per participant #####



### function to combine the taps from block 1 and 2 into single array for each stim per subj
def parseTapsBlocks(dataFrame1, dataFrame2, idealtaps=1600):
    taps_array_1, taps_array_2 = [],[]
    taps_block_1 = dataFrame1
    for tap in taps_block_1:
        taps_array_1.extend(removeStrFormatting(tap))
    #taps_array_1 = taps_array_1.remove('')
    
    taps_block_2 = dataFrame2
    for tap in taps_block_2:
        taps_array_2.extend(removeStrFormatting(tap))
    #taps_array_2 = taps_array_2.remove('')

    alltaps = taps_array_1 + taps_array_2     
    #print('for the taps provided: \n')
    print("%r taps: %r/%r = %r percent" %(csv_file, len(alltaps), idealtaps, 100*len(alltaps)/idealtaps))   
    return (100*len(alltaps)/idealtaps)

#person = 'Randy' # change to plot specific participant 'bb' is A2 which is notimbre3,4 timbre3,4

    
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
            print('subject did not tap to %r' %(sndfile))
            

#%%%  ############################################################################
###########SCORE THE TAPS AND CREATE CSV WITH PARTICIPANT INFO and STATS ######
############################################################################
import csv 
versions = ['A1','A2','B1','B2']

score_cutoff = 80.0 ## only collect subject info from 'usable' batch/data/taps
# string prompts to parse from output pavlovia CSV file  
email_prompt = "If you are not an MTurk worker, please provide your email address"
mturk_prompts = ['If you are a MTurk worker, what is your MTurk worker ID?',"If you are a MTurk worker, what is your MTurk worker ID?'"]
gender_prompt = "What is your sex (f/m/decline)?"
age_prompt = 'What is your age?'
hand_prompt = 'What is your handedness (R/L/ambi)?'
music_training_prompt = 'How many years of musical training, if any, do you have?'
headphones_prompt = 'Are you wearing headphones?'
os_prompt = ['OS']


renamed_participant_csv = []
all_participants = []
stanford_participants = []
csv_files_w_email = {}      # dictionary to hold csv filename indexed by email address
csv_files_w_mturk = {}
subject_tap_score = {}
csv_files_w_win = []
csv_files_w_mac = []
csv_files_w_linux = []
stanford_participants = []
email_given_participants = []

cnt_tag = 1 

import datetime

tap_resp_block1 = 'block1_taps.rt'
tap_resp_block2 = 'block2_taps.rt'
tap_blocks = [tap_resp_block1, tap_resp_block2]

time_tag = '_'.join([str(datetime.datetime.now().month), str(datetime.datetime.now().day), str(datetime.datetime.now().year)])

csv_filename = './psychopy/swarm-tapping-study/mturk-csv/' + batch_folder + '/' + batch_folder + '-' + time_tag + '-participant-STATS.csv'

mtk = []

genders, hands, music_trainings, headphones, ages = [],[],[],[],[]

with open(csv_filename, 'w', newline= '') as file: 
    writer = csv.writer(file)
    writer.writerow(['version', 'day', 'batch', 'participant', 'newname', 'mturk-id','email', 'os-type', 'strong stim score', 'filename'])

    for version in versions:
        for csv_file in glob.glob(os.path.join('./psychopy/swarm-tapping-study/mturk-csv/', batch_folder, version, "*.csv")):
            print('analyzing %r'%(csv_file))
            csvfile = pd.read_csv(csv_file, keep_default_na=False)
            participant_initials = csvfile['Participant Initials'][0]

            
            mturk_id = 'none' 
            email = 'none'
            experiment_version = os.path.basename(csv_file).split('.')[0].split('_')[1].split('-')[1]
            batch = os.path.basename(csv_file).split('.')[0].split('_')[2].split('-')[2]
            month = os.path.basename(csv_file).split('.')[0].split('_')[2].split('-')[1] 
            day = os.path.basename(csv_file).split('.')[0].split('_')[2].split('-')[2]
            month_day = month + '-' + day
            month_day = str(month_day)
            
            score = 0.0
            
            if float(batch) < 16:
                batch_version = '1'
            else:
                batch_version = '2'
            
            try:
                mturk_id = csvfile[mturk_prompts[0]][0]
                email = csvfile[email_prompt][0]
                #print(mturk_id)
            except:
                pass    
            try:
                mturk_id = csvfile[mturk_prompts[1]][0]
                email = csvfile[email_prompt][0]            
                #print(mturk_id)
            except:
                pass
            
            ostype = csvfile['OS'][0]
            if csvfile['OS'][0].startswith('Win'):
                csv_files_w_win.append(csv_file)
            if csvfile['OS'][0].startswith('Mac'):
                csv_files_w_mac.append(csv_file)
            if csvfile['OS'][0].startswith('Lin'):
                csv_files_w_linux.append(csv_file)

            mtk.append(mturk_id)
                    ##############    
            try:
                alltaps_block1 = csvfile[tap_resp_block1][4:44]
                alltaps_block2 = csvfile[tap_resp_block2][44:]
                print('for all taps')
                score = parseTapsBlocks(alltaps_block1, alltaps_block2, idealtaps=1600)
                
                print('for strong stimuli:')
                strong_cond_block1 = csvfile[csvfile.cond.str.contains('strong')][tap_resp_block1]
                strong_cond_block2 = csvfile[csvfile.cond.str.contains('strong')][tap_resp_block2]
                score = parseTapsBlocks(strong_cond_block1.iloc[1:11], strong_cond_block2[11:], idealtaps=400) # ideal  taps 20 beats* (5(tempo_cond)*2(versions)*2(blocks))
            
                subject_tap_score[participant_initials] = score
                
                if score > score_cutoff:
                    gender = csvfile[gender_prompt][0]
                    hand = csvfile[hand_prompt][0]
                    music_training = csvfile[music_training_prompt][0]
                    headphone = csvfile[headphones_prompt][0]
                    age = csvfile[age_prompt][0]
                    
                    genders.append(gender)
                    hands.append(hand)
                    music_trainings.append(music_training)
                    headphones.append(headphone)
                    ages.append(age)
                    
                    ages.append(age)
            
                
                print('----------------------\n')
            except KeyError:
                print('participant %r did not tap'%(participant_initials))
                print('---------------\n')
                subject_tap_score[participant_initials] = 0.0 
                score = 0.0
    
            print(experiment_version, '\t\t', month_day, '\t\t', participant_initials, '\t\t', mturk_id, '\t\t', email, '\t', ostype, '\t\t', os.path.basename(csv_file))
            
            
            participant_renamed = participant_initials + ' ' + str(cnt_tag) + '.csv'
            all_participants.append(participant_renamed)
            
            #make cp of renamed file and --> batch1-renamed dir
            #copyfile(csv_file, os.path.join(renamed_csv_dir, participant_renamed))
            
            renamed_participant_csv.append(participant_renamed)
            
            cnt_tag += 1 # inc counter tag 
            
            writer.writerow([experiment_version, month_day, batch_version, participant_initials, participant_renamed, mturk_id, email, ostype, score, os.path.basename(csv_file)])
#%% ### PARSE TOTAL PARTICIPANT INFO 
newhands = []
num_rh, num_lh, num_ambi, num_NA = 0,0,0,0


fig, ax = plt.subplots(5,1, figsize=(5,10))
# HANDS
for hand in hands:
    if hand.startswith(('R','r')):
        newhands.append('R')
        num_rh += 1
    if hand.startswith(('L','l')):
        newhands.append('L')
        num_lh += 1
    if hand.startswith('a'):
        newhands.append('ambi')
        num_ambi += 1
    if hand.startswith('A'):
        newhands.append('ambi')
        num_ambi += 1   
    # check empty cases
    if not hand:
        newhands.append('NA')
        num_NA += 1

ax[0].hist(newhands, bins=3)
ax[0].set_title('Participants L/R Handedness')

# GENDER 
num_f, num_m, num_NA = 0,0, 0
female, male, not_spec = [],[],[]
for gender in genders:
    if gender.startswith(('F','f')):
        num_f += 1
        female.append('F')
    if gender.startswith(('M','m')):
        num_m += 1
        male.append('M')    
    if not gender.startswith(('F','f','M','m')):
        num_NA += 1
        not_spec.append('NA')
all_genders = female + male + not_spec
ax[1].hist(all_genders, bins=3)



# headphones
num_wearing, num_notwearing = 0,0
wearing, not_wearing = [],[]
for headphone in headphones:
    if headphone.startswith(('Y','y')):
        num_wearing += 1
        wearing.append('wearing')
    if headphone.startswith(('N','n')):
        num_notwearing += 1
        not_wearing.append('not wearing')
all_wearing = wearing + not_wearing
ax[2].hist(all_wearing, bins=2)


import re 
num_musical_training = []   
for mt in music_trainings:
    if str(mt).startswith(('N','n')):
        mt = 0
    print(mt)
    stripped_num = re.sub("[^0-9]", "", str(mt))
    print(stripped_num)
    if mt:
        num_musical_training.append(float(stripped_num))



_ = ax[3].hist(np.array(num_musical_training), bins=30)
ax[3].set_title('Average Years of Musical Training')
ax[3].set_xlabel('avg yrs musical training = %r' %(round(np.nanmean(num_musical_training),2)))
            
            
num_ages = []
for age in ages:
    if age:
        stripped_age = re.sub("[^0-9]", "", str(age))
        num_ages.append(float(stripped_age))
        
print('average age of %r participants: %r'%(len(ages), np.mean(num_ages)))
print('std age of participants is: %r' %(np.std(num_ages)))

print('average years musical training in %r participants: %r'%(len(music_trainings), np.nanmean(num_musical_training)))
print('avg yrs musical training = %r' %(round(np.nanstd(num_musical_training),2)))
    
ax[4].hist(num_ages, bins=30)
ax[4].set_title('Average Age of Participants')
plt.tight_layout()
ax[4].set_xlabel('average age/ SD age = %r / %r' %(np.mean(num_ages), np.std(num_ages)))
plt.savefig('./psychopy/swarm-tapping-study/mturk-csv/' + batch_folder + '/stats/stats.png', dpi=130)     
        
    
    
        
        

#%% DETERMINE HOW MANY USABLE DATASETS FROM EACH VERSION A1/2,B1/2


from shutil import copyfile

df = pd.read_csv(csv_filename)

percent_cutoff_low = 90.0 # participant must score better than 90% on strong stimuli
percent_cutoff_high = 110.0
df_sorted_by_strong = df.sort_values(by='strong stim score')
df_90 = df[df['strong stim score']>percent_cutoff_low]
df_cut = df_90[df_90['strong stim score'] <= percent_cutoff_high]

versions = ['A1','A2','B1','B2']

#old_fi_location = './psychopy/swarm-tapping-study/mturk-csv/' + batch_folder
new_fi_location = './psychopy/swarm-tapping-study/mturk-csv/' + 'usable-' + batch_folder + '/' 

if os.path.exists(new_fi_location) == False:
    os.mkdir(new_fi_location)

usable_subjects = []
for theversion in versions:
    #print(theversion)
    usable_data = df_cut[df_cut['version'] == theversion]['filename']
    print('for', theversion, ':', len(usable_data))   
    
    old_fi_location = os.path.join('./psychopy/swarm-tapping-study/mturk-csv/', batch_folder, theversion)
    for fi in usable_data: 
        old_fi = os.path.join(old_fi_location, fi)
        new_fi = os.path.join(new_fi_location, fi)
        #print(old_fi, new_fi)
        copyfile(old_fi, new_fi)
        
        usable_subjects.append(fi)
        
#%%% count number of stanford participants in usable-batch 
stanford_participants = {'A1': [], 'A2': [], 'B1': [], 'B2': []}
mturk_participants = {'A1': [], 'A2': [], 'B1': [], 'B2': []}

import re
regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'

cnt = 0
for fi in glob.glob(new_fi_location + '*.csv'):
    csvfile = pd.read_csv(fi, keep_default_na=False)
    
    experiment_version = os.path.basename(fi).split('.')[0].split('_')[1].split('-')[1]
    #print(experiment_version)
    email = 'NONE'
    mturk = 'NONE'
    

    #### EMAIL PARSE #######
    try:
        email_str = csvfile[email_prompt][0]
        if email_str.find('@') > 0:
            email = email_str
        else:
            email = 'NONE'
        
        stanford_participants[experiment_version].append(email)
    except:
        pass
    
    ##### MTURK PARSE ####
    try:
        mturk_id = csvfile[mturk_prompts[0]][0]
        mturk_participants[experiment_version].append(mturk_id)
    except:
        mturk_id = 'NONE'
        pass    
    try:
        mturk_id = csvfile[mturk_prompts[1]][0]
        mturk_participants[experiment_version].append(mturk_id)
    except:
        mturk_id = 'NONE'
        pass
        
    print(csvfile['Participant Initials'][0], email, experiment_version,os.path.basename(fi))
    cnt +=1 
print(cnt, ' total participants')
            
#%%
### for standford participants, remove blank text fields (empty strings)
total_email_subjects, total_mturk_subjects = 0, 0
for version in versions:
    stanford_participants[version] = [elem for elem in stanford_participants[version] if elem != '']
    mturk_participants[version] = [elem for elem in mturk_participants[version] if elem != '']
        
    for elem in stanford_participants[version]:
        total_email_subjects += 1
    for elem in mturk_participants[version]:
        total_mturk_subjects += 1
### for mturk participants, remove blank text fields (empty strings)
# total number of stanford participants:
print('STANFORD PARTICIPANT TOTALS:')
for version in versions:
    print('study %r : %r participants'%(version, len(stanford_participants[version])))
print('stanford total: ', total_email_subjects, '\n')

print('MTURK PARTICIPANT TOTALS:')
for version in versions:
    print('study %r : %r participants'%(version, len(mturk_participants[version])))
print('mturk total: ', total_mturk_subjects, '\n')

print('there were %r total participants' %(total_mturk_subjects + total_email_subjects)) 
#%% ########NB::::::: just for mturk usable batch, no need for versions bc all in mturk-batch  

old_fi_location = os.path.join('./psychopy/swarm-tapping-study/mturk-csv/', batch_folder)
for fi in df_90['filename'].values: 
    old_fi = os.path.join(old_fi_location, fi)
    new_fi = os.path.join(new_fi_location, fi)
    #print(old_fi, new_fi)
    copyfile(old_fi, new_fi)
    
    usable_subjects.append(fi)

        

#%% PLOT ALL THE SUBJECTS TAPS ON SINGLE PLOT AND SAVE FIGURE TO DIR of SUBJECT NAME 
thestim2plot = A2_stimuli_names # just stimuli names in 'version'
path2stims = './psychopy/swarm-tapping-study/allstims/'

usable_plot_dir = './psychopy/swarm-tapping-study/mturk-csv/usable-batch/'

for person in usable_subjects:
    print('.... working on %r .....' %(subject[person]))
    plt.figure()
    fig, ax = plt.subplots(len(thestim2plot), 1, sharex=True, sharey=True, figsize=(20,40))
    n = 0
    for j, sndfile in enumerate(sorted(allstims)):
        #print('analyzing', sndfile)
    
        try:
            sync_cond_version = str.split(os.path.basename(sndfile), '.')[0]
            y, sr_ = librosa.load(sndfile)
            subjecttaps = librosa.time_to_samples(subject_resps[person][sync_cond_version]) #need to redefine [(no,timbre)] to look at timbre par exemple
            
            ax[n].plot(y, linewidth=0.5) # plot sound waveform
            ax[n].vlines(subjecttaps, -0.5, 0.5, color='red') # plot subject taps        
            ax[n].vlines(sndbeatbins[sync_cond_version], -0.9, 0.9, color='green') # plot beat bins 
            
            ax[n].set_yticks([]) # turn off y ticks
            ax[n].set_ylabel(sync_cond_version, fontsize=8, rotation=0)
        
            fig.suptitle(person + batch_folder + ' taps ')
            n += 1 
        except KeyError:
            pass
            #print('could not find %r' %(sndfile))
    
    try:
        plt.savefig(os.path.join(usable_plot_dir, 'tap-plots', subject[person] + ' taps.png'), dpi=120)
        plt.close()
    except:
        plt.savefig(os.path.join(usable_plot_dir, 'tap-plots', person + ' taps.png'), dpi=120)
        plt.close()        
    print('\n\n')



#%% ############# ITI of beat sections 

subject_iti_slices = defaultdict(lambda: defaultdict(list))


# NB: need to load the average tempo and then normalize the iti to it 

# no timbre 
for person in usable_subjects:

    subject_iti_slices[person] = {}
    
    for snd in n_strong:
        sync_cond_version = str.split(os.path.basename(snd), '.')[0]
        try: 
            subject_resps[person][sync_cond_version]
            subject_iti_slices[person][sync_cond_version] = []
        except: 
            pass        
        
        for beatseg in beatsegments: 
            try:
                taps = subject_resps[person][sync_cond_version]
                normalized_tap_iti = np.diff(taps)/centerperiods[sync_cond_version]
                tap_iti_slice = normalized_tap_iti[beatseg[0]:beatseg[1]]
                avg_tap_iti_slice = np.mean(tap_iti_slice) 
                
                subject_iti_slices[person][sync_cond_version].append(avg_tap_iti_slice) 
            except: 
                pass

    
    
    
    
    









