#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:46:56 2020

change 'model' and 'version' variable to create pooled arrays from beat section 
txt files generated from check-all-iti_.py script 
this is so that we can apply circular stats in R scripts 
@author: nolanlem
"""


import numpy as np
import glob 
import os 
import matplotlib.pyplot as plt 

os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/analysis-scripts/R-scripts/csv/')
#%%
num_beat_segments = 6
os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/' + str(num_beat_segments) + '-beat-segments/PCs/')
#%%
model = ['model', 'subject'] 
version = ['t','n']

for m in model:
    for v in version:
        strong, medium, weak, none = [],[],[],[]
        
        phases_dir = m + '/phases/' + v
        
        strong_dir = phases_dir + '-strong*.txt'
        medium_dir = phases_dir + '-medium*.txt'
        weak_dir = phases_dir + '-weak*.txt'  
        none_dir = phases_dir + '-none*.txt' 
        
        for txt in glob.glob(strong_dir):
            strong.extend(np.loadtxt(txt))

        for txt in glob.glob(medium_dir):
            medium.extend(np.loadtxt(txt))
                
        for txt in glob.glob(weak_dir):
            weak.extend(np.loadtxt(txt))
        
        for txt in glob.glob(none_dir):
            none.extend(np.loadtxt(txt))
        

        np.savetxt(phases_dir + '-all-strong.txt', np.array(strong), delimiter=',')            
        np.savetxt(phases_dir + '-all-medium.txt', np.array(medium), delimiter=',')            
        np.savetxt(phases_dir + '-all-weak.txt', np.array(weak), delimiter=',')            
        np.savetxt(phases_dir + '-all-none.txt', np.array(none), delimiter=',')            
        
        
        

#%%
    
thefi = os.path.join(dir_to_save, model + '-strong.txt')
np.savetxt(thefi, np.array(strong), delimiter=",") 
thefi = os.path.join(dir_to_save, model + '-medium.txt')
np.savetxt(thefi, np.array(medium), delimiter=",")
thefi = os.path.join(dir_to_save, model + '-weak.txt') 
np.savetxt(thefi, np.array(weak), delimiter=",") 
thefi = os.path.join(dir_to_save, model + '-none.txt')
np.savetxt(thefi, np.array(none), delimiter=",") 
    