#!/usr/bin/env python
# coding: utf-8
from urllib.request import urlopen, urlretrieve
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from subprocess import getoutput as gop
import glob
from re import search
import numpy as np
import csv


class ReconhecimentoPadroes:
    def __init__(self, tamanho):

        urls = {
            'small': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/smni_eeg_data.tar.gz',
            'large_train': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TRAIN.tar.gz',
            'large_test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TEST.tar.gz',
            'full': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/eeg_full.tar'
        }

        self.tamanho = tamanho
        self.data = None
        self.path = tamanho + 'Dataset.csv'

        self.carregar()

    
    def carregar(self):
        # identificando pastas
        folders = {
            self.tamanho: 'dataset/'+self.tamanho,
            'large_train': 'dataset/large_train',
            'large_test': 'dataset/large_test',
            'full': 'dataset/full',
        }
        # carregando pasta do tamanho
        dir = gop('ls {}'.format(folders[self.tamanho])).split('\n')  

        subjects = self.montaDimensoes(folders, dir)

        self.data = np.array(subjects)
        print(self.data.shape)

        self.plotar()


    def montaDimensoes(self, folders, dir):
         # 1ª dimensão dos dados contendo os sujeitos. Ex.: C_1, a_m, etc
        subjects = list()
        for types in dir:
            files = gop('ls {}/{}'.format(folders[self.tamanho], types)).split('\n')
            # 2ª dimensão dos dados contendo as sessões (trials)
            trials = list()
            for f in files:
                arquivo = open('{}/{}/{}'.format(folders[self.tamanho], types, f))
                text = arquivo.readlines()
                # 3ª dimensão dos dados contendo os canais (eletrodos)
                chs = list()
                # 4ª dimensão dos dados contendo os valores em milivolts
                values = list()
                for line in text:
                    # ex: "# FP1 chan 0"
                    t = search('\w{1,3} chan \d{1,2}', line)
                    # ex: "0 FP1 0 -8.921"
                    p = search('^\d{1,2}\ \w{1,3}\ \d{1,3}\ (?P<value>.+$)', line)
                    if p:
                        values.append(float(p.group('value')))
                    # mudou para outro eletrodo
                    elif t and values:
                        chs.append(values)
                        values = list()
                chs.append(values)
                trials.append(chs)
                arquivo.close()
            subjects.append(trials)  
        return subjects

    def montaCSV(self, folders, dir):


        writer = csv.writer(open(self.path, 'w'))
      
       """
       # salvar linhas no csv
        for row in data:
            if counter[row[0]] >= 4:
                writer.writerow(row)
        """
         # 1ª dimensão dos dados contendo os sujeitos. Ex.: C_1, a_m, etc
        subjects = list()
        for types in dir:
            files = gop('ls {}/{}'.format(folders[self.tamanho], types)).split('\n')
            # 2ª dimensão dos dados contendo as sessões (trials)
            trials = list()
            for f in files:
                arquivo = open('{}/{}/{}'.format(folders[self.tamanho], types, f))
                text = arquivo.readlines()
                # 3ª dimensão dos dados contendo os canais (eletrodos)
                chs = list()
                # 4ª dimensão dos dados contendo os valores em milivolts
                values = list()
                for line in text:
                    # ex: "# FP1 chan 0"
                    t = search('\w{1,3} chan \d{1,2}', line)
                    # ex: "0 FP1 0 -8.921"
                    p = search('^\d{1,2}\ \w{1,3}\ \d{1,3}\ (?P<value>.+$)', line)
                    if p:
                        values.append(float(p.group('value')))
                    # mudou para outro eletrodo
                    elif t and values:
                        chs.append(values)
                        values = list()
                chs.append(values)
                trials.append(chs)
                arquivo.close()
            subjects.append(trials)  
        return subjects


    def plotar(self):
        d1 = list()
        d2 = list()

        for e in range(64):
            for i, t in enumerate(np.linspace(0, 1, 256)):
                d1.append([e, t, self.data[0][0][e][i]])
                d2.append([e, t, self.data[1][0][e][i]])
        d1 = np.array(d1)
        d2 = np.array(d2)
        x1, y1, z1 = d1[:,0], d1[:,1], d1[:,2]
        x2, y2, z2 = d2[:,0], d2[:,1], d2[:,2]

        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.plot_trisurf(x1, y1, z1, cmap=cm.inferno, linewidth=1)
        ax.set_xlabel('Canais')
        ax.set_ylabel('Tempo (seg.)')
        ax.set_zlabel('Milivolts')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.plot_trisurf(x2, y2, z2, cmap=cm.inferno, linewidth=1)
        ax.set_xlabel('Canais')
        ax.set_ylabel('Tempo (seg.)')
        ax.set_zlabel('Milivolts')

        fig.colorbar(surf)
        fig.tight_layout()
        plt.show()

    
"""
     ### Continuação...
 
     -+ Melhorar os comentários no códigos
     - Modificar a visualização do gráfico de um trial fixo para a média de todos os trials
         - Fatorar o código o máximo possível (evitar loops desnecessários com o uso de `numpy`
         - Criar mais `subplots` para comparar a visualização
     - Gravar os dados carregados em arquivo(s) CSV de um jeito mais fácil de carregar novamente
     - Fazer o código para os arquivos "large": os arquivos estão divididos em **treino** e **teste** (próximo passo do curso)

     In[ ]:
"""
if __name__ == '__main__':
    ReconhecimentoPadroes('small')
    
