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
os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/generators')

#### INITIALIZE
N = 20
sr_model = 20
sr_audio = 22050

####### set the binaural flag and DISTRIBUTION TYPE (uniform, gaussian) #######
binaural_flag = True # binaural audio or no? 
dist_type = 'gaussian'# which probability density function (uniform, or gaussian)
rootdir = os.path.join('inc-sd/', dist_type, str(N))# hold R, or ramps?  

if binaural_flag == True:
    binaural_str = 'binaural'
else:
    binaural_str = 'mono'
    
makeDir(rootdir)
#%%

# load mono metronome click, non-spatialized 
thesample = '/Users/nolanlem/Documents/kura/kura-python/samples/woodblock_lower.wav'
y, _ = librosa.load(thesample, sr=sr_audio)
y = y*0.5 # reduce amp audiofile 

# load binaural metronome clicks
samples = []
for fi in glob.glob('/Users/nolanlem/Documents/TEACHING/jupyter-nbs/conv-outputs-metronome/*.wav'):
    y_, _ = librosa.load(fi, mono=False)
    samples.append(y_)

seconds = 45.   # length of audio to generate
totalsamps = int(seconds*sr_audio)
audiobuffer_L = np.zeros((N, totalsamps + 4*sr_audio)) # left chan
audiobuffer_R = np.zeros((N, totalsamps + 4*sr_audio)) # right chan

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

def distributeSamples(spatial_flag=True, dist_type = 'uniform'):
    for i in range(N):
        #startpos = beat + gaussian_window[i]
        random_deg = np.random.randint(100)
        y_l = samples[random_deg][0]
        y_r = samples[random_deg][1]
        y_mono = y
        if dist_type == 'uniform':
            startpos = beat + uniform_window[i]
        if dist_type == 'gaussian':
            startpos = beat + gaussian_window[i]
        
        if spatial_flag == True:
            audiobuffer_L[i, startpos:(startpos + len(y_l))] = audiobuffer_L[i, startpos:(startpos + len(y_l))] + y_l
            audiobuffer_R[i, startpos:(startpos + len(y_r))] = audiobuffer_R[i, startpos:(startpos + len(y_r))] + y_r
        
        if spatial_flag == False:
            audiobuffer_L[i, startpos:(startpos + len(y_mono))] = audiobuffer_L[i, startpos:(startpos + len(y_mono))] + y_mono
            audiobuffer_R[i, startpos:(startpos + len(y_mono))] = audiobuffer_R[i, startpos:(startpos + len(y_mono))] + y_mono
                    

def makeAudio(events, iteration, spatial_flag=False):
    
    eventsinsamples = librosa.time_to_samples(events,sr=sr_audio)
    
    # audiobufffers for spatial and mono audio 
    audiobuffer_L = np.zeros(max(eventsinsamples) + 2*sr_audio)
    audiobuffer_R = np.zeros(max(eventsinsamples) + 2*sr_audio)    
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
    audiofi = os.path.join(rootdir, dist_type + '_' + str(N) + '_' + str(np.round(iteration,2)) + '_' + binaural_str + '-' + str(R_target) + '-' + str(width_inc) + '.wav')
    sf.write(audiofi, audio.T, samplerate=sr_audio)
    print('creating', audiofi)
    return audio

def makeDir(dirname):
    if os.path.exists(dirname) == False:
        print('making directory: ', dirname)
        os.mkdir(dirname)
#%%

rate = 1        # 1 Hz , rate to distribute events

totaliters = int(seconds*sr_model)

beats = np.linspace(sr_model, totaliters, sr_model)

freq = 1 # 1 Hz 
period_in_samples = freq*sr_audio
beats_1hz = np.arange(sr_audio, totalsamps, period_in_samples, dtype='int')
g_width = np.linspace(sr_audio, 1, len(beats_1hz))

uniform_width = np.linspace(22050, 1, len(beats_1hz))
trigs = []

period_samps = 22050 # num of samples for 1 Hz isochronous beat where events are distributed aroudn
R_mag_traj, R_ang_traj, width_traj = [], [], []

width_inc = 400 # scalar to +/- width of uniform dist depending on phase coherence 
width_init = period_samps # initialize width of uniform dist to be num of samples at desired freq
uniform_window = np.random.uniform(low=0, high=width_init, size=N)
uniform_window = [int(elem) for elem in uniform_window]

#%%

### INITIALIZE
#R_m, _ = calculateCOP(uniform_window, period_samps)
R_m, _ = calculateCOP(gaussian_window, period_samps, dist_type='gaussian')
R_target = 0.2

    
#%%
w_inc = 0.01
w_init = 2
gaussian_window = np.random.normal(0.5, w_init, size=N)
gaussian_window = [int(elem*sr_audio) for elem in gaussian_window]
gaussian_window = np.array(gaussian_window)

sd_sr = sr_audio*0.37
sd_sec = 0.37
gaussian_window = np.random.normal(0, sd_sec, size=N)
#gaussian_window = [int(elem*sr_audio) for elem in gaussian_window]
gaussian_window = np.array(gaussian_window)

events = []

plt.figure()
fig, ax = plt.subplots(nrows=10, ncols=2, figsize=(5,10), sharex='col', sharey='col')



# movemnt of uniform low and high (l,r) range for uniform distribution
l = np.linspace(-0.1, -1, 10)
r = np.linspace(0.1, 1, 10)
brange = np.arange(1, 10)
# range of Stand Dev for Normal dist. increments upon each iteration 
#sd = np.linspace(0.001,1.0,10)
sd = np.linspace(0.001,0.35,10)

i = 0
for l_, r_, sd_ in zip(l,r,sd):
    events, R_traj = [], []
    for b in brange:
        if dist_type == 'gaussian':
            window = np.random.normal(0, sd_, size=N)
        if dist_type == 'uniform':
            window = np.random.uniform(low=l_, high=r_, size=N)
        b_window = np.array(window)+ b
        events.extend(b_window)
        
        R_m, _ = calculateCOP(window, period_samps, dist_type=dist_type)
        R_traj.append(R_m)
        
    events = np.array(events)
    events = events[events<10.0] # remove events > 10 sec
    events = events[events>0.0]  # remove events < 0 sec
    print('max in events', max(events))
    
    ax[i,0].hist(events, linewidth=0.3, bins=100) ## bins in a way mean what how rhythmic acuity is per second (we can cohere 30 events within a second)
    ax[i,0].vlines(np.arange(1,10), 0, max(events), color='red', linewidth=0.5, alpha=0.5)
    ax[i,0].set_title('SD=' + str(np.round(sd_,2)), fontsize=5)
    
    print('make audio')
    wf = makeAudio(events, sd_, spatial_flag=binaural_flag)
    print('get KDE')
    mx = getKDE(events)
    print('plotting..')
    ax[i,1].plot(mx, linewidth=0.7, color='orange')
    wf_mono = wf[0] + wf[1]
    ax[i,1].plot(wf_mono, linewidth=0.5)
    ax[i,1].vlines(sr_audio*np.arange(1,10),-2,1,color='red', linewidth=0.7, alpha=0.5)
    
    i+=1

plt.tight_layout()
#plt.savefig('./plots/inc-uniform-window.png', dpi=150)
#plt.suptitle(dist_type + ' distribution: sd increments += ' + str(np.mean(np.diff(sd))))
plt.savefig(os.path.join(rootdir, dist_type + '-' + str(N) + '-window-' + str(min(sd)) + '_' + str(max(sd))+ '-sd.png'), dpi=150)

#%%##################################################################
##### get Kernel Density Estimate based off of triggers and PLOT
###################################################################
from scipy.ndimage import gaussian_filter1d



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


    
    
    