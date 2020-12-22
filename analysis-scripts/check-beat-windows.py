#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:59:28 2020

@author: nolanlem
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import librosa 

 
os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/')


fig, ax = plt.subplots(10,1, sharex=True, sharey=True ,figsize=(20,10))
plt.figure()

crand = 255*np.random.random(255)

timbre_flag = False
stim_v = 'stimuli_2'

if timbre_flag == True:
    tflag = 't'
    tdir = 'stim-timbre-5/'
else:
    tflag = 'n'
    tdir = 'stim-no-timbre-5/'


    

for i, audio in enumerate(glob.glob('./' + tdir + stim_v +'/*.wav')):
    if(i < 10):
        basename =os.path.basename(audio).split('.')[0]
        
        allaudiodir = '/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/allstims/'
        newaudioname = basename.split('_')[0] + "_" + tflag + "_" + basename.split('_')[1] + "_" + basename.split('_')[2] + ".wav"
        y, sr_ = librosa.load(allaudiodir + newaudioname)
        y_cutoff = len(y)
        
        beat_windows = np.loadtxt(tdir + stim_v + '/phases/beat-windows/' + basename + '.txt') 
        
        phases = np.load(tdir + stim_v + '/phases/' + basename + '.npy')
        
        for osc in phases[:20,:y_cutoff]:
            v_taps = np.nonzero(osc)[0]
            ax[i].vlines(v_taps, -1,1, linewidth=0.4)
            
        
        ax[i].plot(y)
        ax[i].vlines(beat_windows, -1,1, color='red')
        ax[i].set_title(audio)

plt.savefig('/Users/nolanlem/Desktop/bw_.png', dpi=160)
        