#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:56:32 2020

@author: nolanlem
"""

import numpy as np 
import matplotlib.pyplot as plt 
import soundfile as sf
import os
import librosa
import glob
from scipy.interpolate import interp1d
os.chdir('/Users/nolanlem/Documents/kura/kura-git/swarm-tapping-study/generators')
from util.utils import makeDir
import seaborn as sns 
from scipy.ndimage import gaussian_filter1d

sns.set()
sns.set_palette('tab10')

sr_audio = 22050 
# load mono metronome click, non-spatialized 
thesample = './sampleaudio/woodblock_lower.wav'
y, _ = librosa.load(thesample, sr=sr_audio)
y = y*0.5 # reduce amp audiofile 

###### load binaural metronome clicks
samples = []
for fi in glob.glob('./sampleaudio/binaural-metros/*.wav'):
    y_, _ = librosa.load(fi, mono=False)
    samples.append(y_)

# get largest sample in samples array 
tmpmax = samples[0][0].shape[0]
for deg in range(len(samples)):
    maxnum = samples[deg][0].shape[0]
    if maxnum > tmpmax:
        tmpmax = maxnum
largestsampnum = tmpmax

# PER STIMULUS TIME HISTORGRAM 
def PSTH(x, N, len_y):
    # x = spikes/taps matrix 
    # N = 0.1 seconds = 0.1*22050 = 2205 samples
    spike_blocks = np.linspace(0, x.shape[1], int(len_y/N))
    spike_blocks_int = [int(elem) for elem in spike_blocks]
    mx = []
    for i in range(1, len(spike_blocks_int)):
        tapblock_mx = np.nanmean(spikes[:, spike_blocks_int[i-1]:spike_blocks_int[i]])
        block_mx = tapblock_mx*np.ones(spike_blocks_int[i] - spike_blocks_int[i-1])
        mx.extend(block_mx)
    return mx 

def calculateCOP(window, period_samps, dist_type = 'uniform'):
    if dist_type == 'uniform':
        window = [elem for elem in window if elem <= period_samps]
    if dist_type == 'gaussian':
        window = np.array(window) + int(period_samps/2)
        window = [elem for elem in window if elem <= period_samps and elem >= 0]
        
    bininterp = interp1d([0, period_samps], [0, 2*np.pi])
    phases = bininterp(window)   
    R = np.nansum(np.exp(phases*1j))/N
    R_mag = np.abs(R)
    R_ang = np.angle(R)
    R_mag_traj.append(R_mag)
    R_ang_traj.append(R_ang)
    return R_mag, R_ang
        

def makeAudio(events, iteration, stimdir, spatial_flag=False):   
    eventsinsamples = librosa.time_to_samples(events,sr=sr_audio)   
    # audiobufffers for spatial and mono audio     
    audiobuffer_L = np.zeros(max(eventsinsamples) + largestsampnum)
    audiobuffer_R = np.zeros(max(eventsinsamples) + largestsampnum)    
    y_mono = y
    
    for startpos in eventsinsamples:
        random_deg = np.random.randint(100)
        y_l = samples[random_deg][0]
        y_r = samples[random_deg][1]

        if spatial_flag == True:
            audiobuffer_L[startpos:(startpos + len(y_l))] = audiobuffer_L[startpos:(startpos + len(y_l))] + y_l
            audiobuffer_R[startpos:(startpos + len(y_r))] = audiobuffer_R[startpos:(startpos + len(y_r))] + y_r
        
        if spatial_flag == False:
            audiobuffer_L[startpos:(startpos + len(y_mono))] = audiobuffer_L[startpos:(startpos + len(y_mono))] + y_mono
            audiobuffer_R[startpos:(startpos + len(y_mono))] = audiobuffer_R[startpos:(startpos + len(y_mono))] + y_mono

    #audio_l = np.sum(audiobuffer_L, axis=0)
    #audio_r = np.sum(audiobuffer_R, axis=0)
    
    audio_l = 0.8*audiobuffer_L/max(audiobuffer_L)
    audio_r = 0.8*audiobuffer_R/max(audiobuffer_R)
    
    audio = np.array([audio_l, audio_r])
    audiofi = os.path.join(stimdir, dist_type[0] + '_' + binaural_str[0] + '_' + str(N) + '_' + iteration + '.wav')
    sf.write(audiofi, audio.T, samplerate=sr_audio)
    print('creating', audiofi)
    return audio

def getKDE(events):
    eventsinsamples = librosa.time_to_samples(events)
    taps = np.zeros(max(eventsinsamples)+1)
    
    for spike in eventsinsamples:
        np.put(taps, spike, 1)
    blocksize = int(sr_audio/10)
    blocks = np.arange(0, len(taps), blocksize)
    mx = []
    
    for j in range(1, len(blocks)):
        tapblock_mx = np.mean(taps[blocks[j-1]:blocks[j]])
        block_mx = tapblock_mx*np.ones(blocksize)
        mx.extend(block_mx)
        
    gmx = gaussian_filter1d(mx, 1000)
    gmx = gmx/max(gmx)
    gmx -= 2 # move it down below wf amplitude space 
    
    return gmx

def removeAudioStims(thestimdir):
    for folder in thestimdir:
        if os.path.exists(folder):
            for fi in glob.glob(folder + "/*.wav"):
                os.remove(fi)
            for png in glob.glob(folder + '/*.png'):
                os.remove(png)

# util function for
def round2dec(num2round):
    roundednum = np.round(num2round,2)
    return roundednum
#%%

    
b, _ = np.histogram(events, bins=len(events))
mx = []
bsize = 10
for j in range(bsize, len(events), bsize):
    tb_mx = np.mean(b[(j-bsize):j])
    block_mx = tb_mx*np.ones(bsize)
    mx.extend(block_mx)
plt.plot(mx)



#%%##########################################
########## INITIALIZE GLOBAL PARAMS #########
###########################################
N = 40
sr_model = 20
sr_audio = 22050

####### set the binaural flag and DISTRIBUTION TYPE (uniform, gaussian) #######
binaural_flag = True # binaural audio or no? 
dist_type = 'gaussian'# which probability density function (uniform, or gaussian)
if binaural_flag == True:
    binaural_str = 'binaural'
else:
    binaural_str = 'mono'

#### LOOP PARAMS
target_tempos = np.geomspace(start=60, stop=100, num=5)
freq_conds = target_tempos/60. # tempo to distribute events around
period_conds = 1/freq_conds 
num_beats = 15  # number of beats
seconds = (1./freq_conds)*num_beats   # length of audio to generate 
beg_delay = 0.5 # time (secs) to insert in beginning of stim audio
end_delay = 0.5 # time to insert at end fo audio
period_samps = np.array(sr_audio*1./freq_conds, dtype=np.int) # num of samples for 1 Hz isochronous beat where events are distributed aroudn

totalsamps = beg_delay*sr_audio + seconds*sr_audio + end_delay*sr_audio
totalsamps = totalsamps.astype(int)
totalsecs = totalsamps/sr_audio

stimdirs = []

for tmp in target_tempos:
    rootdir = os.path.join('stim-step/' + str(N) + '_' + binaural_str, str(int(tmp))) # hold R, or ramps?  
    stimdirs.append(rootdir)
    makeDir(rootdir)    # make dir for storing stims 

########## RANGE of UNIFORM low and high (l,r) range for uniform distribution
num_grads = 8   # number of SD gradations for each tempo cond

l = np.linspace(-0.1, -1, num_grads)
r = np.linspace(0.1, 1, num_grads)
#brange = np.linspace(1, num_beats, num_beats)
#brange_secs = np.linspace(0.1, 0.1 + num_beats*1./period_conds, num_beats)

####### RANGE of Stand Dev for NORMAL dist. increments upon each iteration 
#sd = np.linspace(0.001,1.0,10)
sd = np.linspace(0.1,0.35,num_grads)    # range of SDs to increment through to make stims
sd_start = 0.4                                  # starting SD, wide enough for no synchrony 
sd_step_targets = np.linspace(0.1, 0.3, num_grads) # step targets to change SD to after starting_beats 
removeAudioStims(stimdirs)
#%%
import scipy.stats as stats

mu = 1
variance = 0.4
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, N)
#plt.plot(x, stats.norm.pdf(x, mu, sigma))
norm_pdf = stats.norm.pdf(x, mu, sigma)
window = np.random.normal(1, 0.37, size=N)
plt.hist(window, bins=20, density=True)
plt.plot(x,norm_pdf)
plt.show()

#%%
sigma = 0.4
mu = 0
window = np.random.normal(mu, sigma, size=N)
count, bins, ignored = plt.hist(window, bins=30, density=True)
pdfcont = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(bins-mu)**2 / (2*sigma**2))
plt.plot(bins, pdfcont, linewidth=2, color='r')

#%%
def returnSmoothPDF(window):
    count, bins, ignored = plt.hist(window, bins=N)
    pdfcont = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(bins-mu)**2 / (2*sigma**2))
    return pdfcont

    
#%% PLOT PDF of GAUSSIAN DISTRIBUTION AT STEP CHANGE 

sd = np.linspace(0.1,0.35,num_grads)    # range of SDs to increment through to make stims
sd_start = 0.4                               # starting SD, wide enough for no synchrony 
sd_step_targets = np.linspace(0.1, 0.3, num_grads) # step targets to change SD to after starting_beats 

mu = 0
fig, ax = plt.subplots(nrows=(num_grads), ncols=1, figsize=(10,13), sharex=True, sharey=True)


for n, sd_target in enumerate(sd_step_targets):
    buffers = np.zeros((num_beats, int(brange_secs[-1]*N+N)))
    trigs = np.zeros((num_beats,int(brange_secs[-1]*N+N)))
    events, R_traj = [], []
    
    
    startbeats = np.random.randint(low=3, high=6)   # number of beats to wait until to apply SD step change
    sd_traj = [sd_start for elem in range(startbeats)] # fill up SD trajectory (per beat) with starting SDZ
    sd_traj.extend([sd_target for elem in range(num_beats - startbeats)]) # fill up with SD target 

    i = 0
    for b, sd_ in zip(brange_secs, sd_traj):
        x = np.linspace(mu - 3*sd_, mu + 3*sd_, N)
        norm_pdf = stats.norm.pdf(x, mu, sd_)
        loc = int(b*N)
        buffers[i, loc:(loc+N)] = norm_pdf
        
        window = np.random.normal(0, sd_, size=N)
        norm_vals = stats.norm.rvs(loc=mu, scale=sd_, size=N)      
        #trigs[i, loc:(loc+N)] = N*(norm_vals + b)
        trigsecs = norm_vals + b
        events.extend(trigsecs)
        i+=1
    
    events = np.array(events)
    events_N = np.array(N*events, dtype=np.int)
    minevents = int(np.abs(min(events_N[events_N<0])))
    events_N = events_N + int(N/2)
    #events_N = events_N[:int(brange_secs[-1]*N+N)]

    for buffer, tr in zip(buffers, trigs):
        ax[n].plot(buffer)
    ax[n].vlines(events_N, -2, -1, linewidth=0.5, alpha=0.6)
    ax[n].set_title('SD ' + str(round2dec(sd_traj[0])) + '->' + str(round2dec(sd_traj[-1])))

    ymin, ymax = ax[n].get_ylim()
    ax[n].vlines(int(brange_secs[1]*N + startbeats*N/2), 0, ymax, color='green', linestyle='--')

plt.tight_layout()
fig.suptitle('Audio Onset Events PDFs with step changes')    
        
        
        

#%%
removeAudioStims(stimdirs)
R_mag_traj, R_ang_traj, width_traj = [], [], []
events, trigs = [], []

plt.figure()
for tmp, stimdir, totalsec, period_samp in zip(freq_conds, stimdirs, totalsecs, period_samps):
    
    fig, ax = plt.subplots(nrows=len(sd_step_targets), ncols=2, figsize=(10, 6), sharex='col', sharey='col')
    fig_r, ax_r = plt.subplots(1,1)
        
    brange_secs = np.linspace(beg_delay, num_beats*1./tmp, num_beats)    
    
    i = 0
    for sd_target in sd_step_targets:
        events, R_traj = [], []
        
        startbeats = np.random.randint(low=3, high=6)   # number of beats to wait until to apply SD step change
        sd_traj = [sd_start for elem in range(startbeats)] # fill up SD trajectory (per beat) with starting SDZ
        sd_traj.extend([sd_target for elem in range(num_beats - startbeats)]) # fill up with SD target 
        
        # distribute gaussian window over beats 
        for b, sd_ in zip(brange_secs, sd_traj):
            if dist_type == 'gaussian':
                window = np.random.normal(0, sd_, size=N)
            b_window = np.array(window)+ b # distribute prob events window around beats depending on tempo (in sec)
            events.extend(b_window)
            
            R_m, _ = calculateCOP(window, period_samp/sr_audio, dist_type=dist_type)
            R_traj.append(R_m)
            
        events = np.array(events)
        events = events[events < totalsec] # remove events > max sec
        events = events[events > 0.0]  # remove events < 0 sec
        beatlocations = np.linspace(beg_delay, (1./tmp)*num_beats, num_beats)

        ax[i,0].hist(events, linewidth=0.3, bins=100) ## bins in a way mean what how rhythmic acuity is per second (we can cohere 30 events within a second)
        beatlocations = np.linspace(beg_delay, (1./tmp)*num_beats, num_beats)
        ax[i,0].vlines(beatlocations, 0, 18, color='red', linewidth=0.5, alpha=0.5)
        ax[i,0].vlines(startbeats, 0, 18, color='green', linewidth=1.5, linestyle='--') # beat where step happens
        iteration_str = str(round2dec(sd_traj[0])) + '->' + str(round2dec(sd_traj[-1])) + '_' + str(startbeats)
        ax[i,0].set_title('SD=' + iteration_str, fontsize=10)
        
        print('making audio')
        wf = makeAudio(events, iteration_str, stimdir, spatial_flag=binaural_flag)
        print('get KDE')
        mx = getKDE(events)
        print('plotting KDE..')
        ax[i,1].plot(mx, linewidth=0.7, color='orange')
        
        wf_mono = wf[0] + wf[1]
        ax[i,1].plot(wf_mono, linewidth=0.5)
        beatlocs_samps = librosa.time_to_samples(beatlocations)
        ax[i,1].vlines(beatlocs_samps, -2, 1, color='red', linewidth=0.5, alpha=0.5, zorder=1)          
        ax[i,1].plot(beatlocs_samps, R_traj, linewidth=1, color='red')
        ax[i,1].vlines(int(startbeats*sr_audio), -2, 1, color='green', linewidth=1.5, linestyle='--') # beat where step happens
        
        #ax[i,1].annotate(str(round2dec(R_traj[-1])), xy=(beatlocs_samps[-1], R_traj[-1]), color='red', fontsize=8)

        # R comparisons figure
        ax_r.plot(beatlocations, R_traj, linewidth=1)
        #ax_r.annotate(str(round2dec(R_mean)), xy=(beatlocations[-1], R_traj[-1]), color='red', fontsize=8)
       
        i+=1
    ax_r.set_ylim([0,1])
    ax_r.set_title(str(round2dec(60/(1/tmp))) + ' phase coherence magnitudes |R| Comparison')
    fig_r.savefig(os.path.join(stimdir, 'R' + '_' + dist_type[0] + '_' + binaural_str[0] + '-' + str(N) + '-' + iteration_str + '.png'))
    # remove x-ticks on right col
    for ax_ in ax[:,1].flat:
        currxticks = ax_.get_xticks()
        ax_.set_xticklabels(np.round(currxticks/sr_audio,1))
        ax_.set_yticklabels([])
    
    fig_r.suptitle(str(round2dec(60/(1/tmp))) + " Hz", fontsize=12)
    fig.suptitle(str(round2dec(60/(1/tmp))) + " Hz", fontsize=12)
    fig_r.tight_layout()
    fig.tight_layout()
    #plt.savefig('./plots/inc-uniform-window.png', dpi=150)
    #plt.suptitle(dist_type + ' distribution: sd increments += ' + str(np.mean(np.diff(sd))))
    fig.savefig(os.path.join(stimdir, dist_type[0] + '_' + binaural_str[0] + '-' + str(N) + '-' + iteration_str + '.png'), dpi=150)



#%%##################################################################
##### get Kernel Density Estimate based off of triggers and PLOT
###################################################################

plt.figure()
fig, ax = plt.subplots(2,2)
for ax_ in ax[:,1].flat:
    



    
#%%

n = 0
for beat, width, width_n in zip(beats_1hz, g_width, uniform_width):
    print('iter:', n, 'R:', R_m, 'w:', w_init)
    # uncomment if you wanna use a gaussian window 
    # gaussian_window = np.random.normal(0, int(width), N)
    # gaussian_window[gaussian_window < -sr_audio] = -sr_audio
    # gaussian_window = [int(elem) for elem in gaussian_window]
   
    # uniform distribution that decreases width
    # uniform_window = np.random.uniform(low=0, high=width_n, size=N)
    # uniform_window = [int(elem) for elem in uniform_window]
    
    # ADAPTIVE GAUSSIAN WINDOW 
    #R_m, R_a = calculateCOP(gaussian_window, period_samps, dist_type=dist_type)

    # ADAPTIVE uniform distribution that tries to keep R at R_target
    R_m, R_a = calculateCOP(uniform_window, period_samps)
   
    if R_m < R_target:
        width_init -= width_inc
        uniform_window = np.random.uniform(low=0, high=width_init, size=N)
        
        w_init -= w_inc
        gaussian_window = np.random.normal(0, w_init, size=N)
    if R_m >= R_target:
        width_init += width_inc 
        uniform_window = np.random.uniform(low=0, high=width_init, size=N)
        
        w_init += w_inc
        gaussian_window = np.random.normal(0, w_init, size=N)

    width_traj.append(width_init)
    
    # reinit distribution windows
    gaussian_window = [int(elem*sr_audio) for elem in gaussian_window]
    uniform_window = [int(elem) for elem in uniform_window]
         
    trigs.extend(uniform_window + beat)
    
    distributeSamples(binaural_flag, dist_type)

    n+=1
    
    
audio_l = np.sum(audiobuffer_L, axis=0)
audio_r = np.sum(audiobuffer_R, axis=0)

audio_l = 0.8*audio_l/max(audio_l)
audio_r = 0.8*audio_r/max(audio_r)

audio = np.array([audio_l, audio_r])
audiofi = rootdir + dist_type + '_' + str(N) + '_' + binaural_str + '-' + str(R_target) + '-' + str(width_inc) + '.wav'
sf.write(audiofi, audio.T, samplerate=sr_audio)
print('creating', audiofi)
plt.plot(R_mag_traj)
plt.suptitle('Phase Coherence |R| Trajectory')
plt.savefig(rootdir + os.path.basename(audiofi)[0] + '.png', dpi=150)

#%%
##################################################################
##### get Kernel Density Estimate based off of triggers and PLOT
###################################################################
from scipy.ndimage import gaussian_filter1d

def getKDE(events):
    eventsinsamples = librosa.time_to_samples(events)
    taps = np.zeros(max(eventsinsamples)+1)
    
    for i, spike in enumerate(eventsinsamples):
        np.put(taps, spike, 1)
    blocksize = int(sr_audio/10)
    blocks = np.arange(0, len(taps), blocksize)
    mx = []
    
    for i in range(1, len(blocks)):
        tapblock_mx = np.mean(taps[blocks[i-1]:blocks[i]])
        block_mx = tapblock_mx*np.ones(blocksize)
        mx.extend(block_mx)
     
    gmx = gaussian_filter1d(mx, 1000)
    gmx = gmx/max(gmx)
    gmx -= 2 # move it down below wf amplitude space 
    
    return gmx
    
    #y, sr_ = librosa.load(audiofi) # plot waveform 
    #plt.plot(y)
    #plt.plot(gmx, linewidth=0.5)
    #plt.savefig(rootdir + os.path.splitext(os.path.basename(audiofi))[0] + '.png', dpi=150)


#%%%%

taps = np.zeros(totalsamps + 4*sr_audio)

#spikes = np.zeros((tap_mat.shape[0], 1 + int(np.nanmax(tap_mat))))
# for j, taps in enumerate(tap_mat):
#     #non_nan_taps = taps[np.logical_not(np.isnan(taps))]
#     #non_nan_taps = [int(tap) for tap in non_nan_taps]
#     np.put(spikes[j], non_nan_taps, 1.0)
    
for i, spike in enumerate(trigs):
    np.put(taps, trigs[i], 1)

blocksize = int(sr_audio/10)
blocks = np.arange(0, len(taps), blocksize)
mx = []

for i in range(1, len(blocks)):
    tapblock_mx = np.mean(taps[blocks[i-1]:blocks[i]])
    block_mx = tapblock_mx*np.ones(blocksize)
    mx.extend(block_mx)
 
gmx = gaussian_filter1d(mx, 1000)
gmx = gmx/max(gmx)
gmx -= 2 # move it down below wf amplitude space 

y, sr_ = librosa.load(audiofi) # plot waveform 
plt.plot(y)
plt.plot(gmx, linewidth=0.5)
plt.savefig(rootdir + os.path.splitext(os.path.basename(audiofi))[0] + '.png', dpi=150)


#%%   
plt.figure()
for i, b in enumerate(beats_1hz):
    for pt in uwindow[i]:
        plt.vlines(pt,-1,1)
#%%
mx = PSTH(np.array(spikes).T, 1102, len(y_[0])) # per stimulus time histogram, 2205 = 0.1 seconds 
plt.plot(mx)
#%%


    
    
    