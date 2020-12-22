#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:35:13 2020

@author: nolanlem
"""


os.chdir('/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/csvs/')






#%% 
df = pd.read_csv('/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/csvs/10-13_16-43.csv')

df = df.dropna()

df.to_csv('/Users/nolanlem/Documents/kura/kura-new-cond/py/psychopy/swarm-tapping-study/analysis-scripts/plots/beat-segment-analysis/6-beat-segments/csvs/out-new.csv', index=False)

