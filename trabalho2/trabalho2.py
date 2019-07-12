#!/usr/bin/env python
# coding: utf-8

# # Trabalho 2
# 

# Objetivo: ALPHA

from mne.time_frequency import psd_welch
import pandas as pand
from mne.time_frequency import psd_welch as pw
import numpy as np
import mne
import matplotlib.pyplot as plt


ch_names_1 = ['index', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'time']
ch_names_utils = ['PO3', 'PO4', 'P8', 'O1', 'O2', 'P7']
buffer_size = 5

def read_remove_eletro(nameData):
    readed = pand.read_csv(nameData, skiprows=6, names=ch_names_1)
    readed = readed.drop(['index','7', '8', '10', '11', '12', 'time'], axis=1)
    len_readed = len(readed)
    return len_readed, readed.transpose()

# Cria transforma o domínio para frequência e aplica o filtro passa faixa 4x
def aux_pre_proc(data):

    ch_types = ['eeg'] * 6

    info = mne.create_info(ch_names=ch_names_utils, sfreq=256, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose= False)
    mont = mne.channels.read_montage('standard_1020')
    raw.set_montage(mont)

    raw.notch_filter(np.arange(60,121,60), fir_design='firwin')

    raw.filter(5,50)
    raw.filter(5,50)
    raw.filter(5,50)
    raw.filter(5,50)

    return raw

def plot_graphics_frequency(avg_freq):
    
    freq = avg_freq
    y = freq.keys()
    x = freq.values()
    
    plt.bar(y, x)
    
    plt.pause(1.00)
    plt.clf()

def routine(results):
    plot_graphics_frequency(results)

    results_sorted = sorted(list(results.values()), reverse=True)

    print('Média: '+str(100*(results_sorted[0] - results_sorted[1])/results['alpha']))
    
def exec_thread(raw, len_readed):
    # estrutura para armazenar os valores para cada tipo de eletrodo
    results = {
        'beta': {},
        'alpha': {},
        'gamma': {},
        'theta': {}
    }
    # estrutura para armazenar os intervalos de frequencia para cada eletrodo
    interval_frenquency = {
        'beta': { 'begin': 12, 'end': 30 },
        'alpha': { 'begin': 8, 'end': 12 },
        'gamma': { 'begin': 25, 'end': 100 },
        'theta': { 'begin': 5, 'end': 7 }
    }
    for i in range(0, int(len_readed/256)):

        # deslocamento dado um buffer pre-definido a cada 1s
        psdw, freq = psd_welch(raw, fmin=5, fmax=50, tmin=i, tmax=i+buffer_size, verbose=False)
        for j in results.keys():
            results[j] = max(
                            np.mean(
                                psdw[:,interval_frenquency[j]['begin']:interval_frenquency[j]['end']],
                                axis=1
                            )
                        )
        if (max(results.values()) == results['alpha']):
            routine(results)

if __name__ == "__main__":
    len_readed, readed = read_remove_eletro('teste.csv')
    processed = aux_pre_proc(readed)
    exec_thread(processed, len_readed)