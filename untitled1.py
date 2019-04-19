# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:27:57 2019

@author: Deepa.Amasar
"""
import matplotlib.pyplot as plt
import numpy as np
import wave

spf = wave.open('OSR_us_000_0010_8k.wav','r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()

silenceArrStart = []
silenceArrEnd = []
first = False
threshold = 2500
for i in range(signal.shape[0]-1):
    if( signal[i]<threshold and signal[i]>-threshold):        
        signal[i]=0
        if(first == False):
            silenceArrStart.append(i/fs)
            first = True        
        if(signal[i+1]>threshold or signal[i+1]<-threshold):
            
            silenceArrEnd.append(i/fs)
            first = False
Time=np.linspace(0, len(signal)/fs, num=len(signal))
plt.figure(1)
plt.title('Signal Wave...')
plt.plot(Time,signal)
plt.show()

for i in range(len(silenceArrStart)):
    diff = silenceArrEnd[i] - silenceArrStart[i]
    if(diff>1):
        print( silenceArrStart[i],silenceArrEnd[i] )
    
    
    
    
    