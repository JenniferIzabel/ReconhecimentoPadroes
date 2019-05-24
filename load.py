#!/usr/bin/env python
# coding: utf-8

# # Carregamento e preparação de *datasets*
# 
# O carregamento e preparação de *datasets* é um ótimo exercício para tomarmos conhecimento das ferramentas a serem utilizadas para o processamento de sinais em `python`, seja sinais biológicos quanto de outra natureza, como um som, corrente elétrica, etc.
# 
# Nesta `notebook` será apresentado o carregamento de um *dataset* público do *website* `UCI - Machine Learning Repository`. O *dataset* a ser utilizado é o `EEG Database Data Set` (https://archive.ics.uci.edu/ml/datasets/EEG+Database).
# 
# 
# ## Descrição do *dataset*:
# 
# A intenção deste *dataset* é examinar por meio de algoritmos de inteligência computacional a pré-disposição genética que um paciente possui ao alcoolismo.
# 
# Os principais dados analizados são do tipo *time-series*, em outras palavras, conjuntos de dados que representam um sinal mensurado no domínio do tempo. Os dados são completados com outros atributos como o nome do eletrodo, o número da amostra, etc. Outras informações relevantes do *dataset*:
# 
# - Quantidade de atributos: 4
# - Número de instancias: 122
# - Existem dados faltantes? Sim
# - Tipos de dados encontrados: categórico, inteiro e real
# 
# Existem três categorias de dados neste *dataset*:
# 
# 1. Small Data Set: <font color='red'>**descrever**</font>
# 2. Large Data Set: <font color='red'>**descrever**</font>
# 3. Full Data Set: <font color='red'>**descrever**</font>
# 
# Cada sessão (*trial*) é armazenada da seguinte forma:
# 
# ```
# # co2a0000364.rd 
# # 120 trials, 64 chans, 416 samples 368 post_stim samples 
# # 3.906000 msecs uV 
# # S1 obj , trial 0 
# # FP1 chan 0 
# 0 FP1 0 -8.921 
# 0 FP1 1 -8.433 
# 0 FP1 2 -2.574 
# 0 FP1 3 5.239 
# 0 FP1 4 11.587 
# 0 FP1 5 14.028
# ...
# ```
# 
# As primeiras 4 linhas são de cabeçalho:
# 
# **linha 1**: identificação do paciente e se ele indica ser um alcoólatra (a) ou controle (c) pela quarta letra (co2**a**0000364);
# 
# **linha 4**: determina se o paciente foi exposto a um único estímulo (`S1 obj`), a dois estímulos iguais (`S2 match`) ou a dois estímulos diferentes (`S2 no match`);
# 
# **linha 5**: identifica o início da coleta dos dados pelo eletrodo FP1. As 4 colunas são:
# 
# ```
# número_da_sessão identificação_do_eletrodo número_da_amostra valor_em_micro_volts
# ```
# 
# 
# ### Realizando o download 
# 
# Primeiro faremos um código para verificar se o *dataset* já foi baixado, caso contrário, executar o código de download:

# In[1]:
import random

from subprocess import getoutput as gop
import glob

from urllib.request import urlopen, urlretrieve
import os

from re import search
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.metrics import confusion_matrix, accuracy_score

urls = {
    'small': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/smni_eeg_data.tar.gz',
    'large_train': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TRAIN.tar.gz',
    'large_test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TEST.tar.gz',
    'full': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/eeg_full.tar'
}

# identificando pastas
folders = {
    'small': 'dataset/small',
    'large_train': 'dataset/large_train',
    'large_test': 'dataset/large_test',
    'full': 'dataset/full',
}


# verifica se o diretório dos datasets existe, se não, baixa a base
def verifica_diretorio():
    if not os.path.exists('dataset/'):
        os.mkdir('dataset/')
        for k, v in urls.items():
            fn = v.split('/')[-1]
            print('Baixando:', fn, '...')
            urlretrieve(v, './dataset/{}'.format(fn))
        print('Downlod dos datasets concluído!')
    else:
        print('Dataset já baixado!')


# ### Descompactando pastas e subpastas
# 
# Agora é necessário descompactar (recursivamente) diversas pastas e subpastas em arquivos GZip. Algumas pastas estão com o arquivo na extensão `.tar`, já outras, `.tar.gz`. Não obstante, algumas subpastas estão compactadas e outras não.

# In[2]:


# único arquivo somente empacotado (tar)
def descompactar_base():
    os.mkdir('dataset/eeg_full/')
    gop('tar -xvf dataset/eeg_full.tar -C dataset/eeg_full')
    os.remove('dataset/eeg_full.tar')

    while glob.glob('dataset/**/*.gz', recursive=True):
        # quando o arquivo está empacotado (tar) e compactado (gz)
        for f in glob.iglob('dataset/**/*.tar.gz', recursive=True):
            gop('tar -zxvf {} -C {}'.format(f, f[:f.rindex('/')]))
            os.remove(f)
        # quando o arquivo está somente compactado (gz)
        for f in glob.iglob('dataset/**/*.gz', recursive=True):
            gop('gzip -d {}'.format(f))
    print('Descompactações finalizadas!')


# ### Carregando parte do dataset
# 
# Vamos agora carregar o subconjunto "small" do *dataset* e fica como <font color='red'>**tarefa de casa**</font> carregar e preparar todos os outros subconjuntos...

# In[3]:


# organizando melhor as pastas
def organiza():
    os.rename('dataset/smni_eeg_data', 'dataset/small')
    os.rename('dataset/eeg_full', 'dataset/full')
    os.rename('dataset/SMNI_CMI_TRAIN/', 'dataset/large_train/')
    os.rename('dataset/SMNI_CMI_TEST/', 'dataset/large_test/')
    print(gop('ls -l dataset/'))


# In[4]:



# carregando pasta "small"
def carrega_small():
    diretory = gop('ls {}'.format(folders['small'])).split('\n')
    # 1ª dimensão dos dados contendo os sujeitos. Ex.: C_1, a_m, etc
    subjects = list()
    for types in diretory:
        files = gop('ls {}/{}'.format(folders['small'], types)).split('\n')
        # 2ª dimensão dos dados contendo as sessões (trials)
        trials = list()
        for f in files:
            arquivo = open('{}/{}/{}'.format(folders['small'], types, f))
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
    data = np.array(subjects)
    print(data.shape)
    return data


# ### Dados carregados...
# 
# Os dados "single" foram dividos da seguinte forma:
# ```
# [experimentos, triagens, canais, amostras]
# ```
# formando um `numpy.array` de quatro dimensões.
# 
# Em seguida, vamos plotar esses dados para "tentar" visualizar algum padrão.

# In[5]:

def plot_dados(data):
    #get_ipython().run_line_magic('matplotlib', 'inline')


    d1 = list()
    d2 = list()

    for e in range(64):
        for i, t in enumerate(np.linspace(0, 1, 256)):
            d1.append([e, t, data[0][0][e][i]])
            d2.append([e, t, data[1][0][e][i]])
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
    #plt.show()

def carregar_exemplo():
    #verifica_diretorio()
    #descompactar_base()
    #organiza()
    
    data = carrega_small()

    plot_dados(data)
################################################################################################################################################################
# DEV TO READ THE TRAIN AND TEST
################################################################################################################################################################
def get_all_datas(files, subjects, path_files, types):
    trials = list()
    for f in files:
        arquivo = open('{}/{}/{}'.format(folders[path_files], types, f))
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


def load_bases2(path_files):
    diretory = gop('ls {}'.format(folders[path_files])).split('\n')

    subA = list()
    subC = list()

    for types in diretory:
        files = gop('ls {}/{}'.format(folders[path_files], types)).split('\n')
        if 'Co2A' in types.title():
            get_all_datas(files, subA, path_files, types)
        else:
            get_all_datas(files, subC, path_files, types)

    return [subA, subC]


def pre_pros(data):
    eletrodosA = list()
    eletrodosC = list()
    for pasta in data[0]:
        for trial in pasta:
            for eletrodo in trial:
                eletrodosA.append(eletrodo)
    
    for pasta in data[1]:
        for trial in pasta:
            for eletrodo in trial:
                eletrodosC.append(eletrodo)

    classes = list()

    for i in range(0, len(eletrodosA)):
        classes.append(1)
    for i in range(0, len(eletrodosC)):
        classes.append(0)

    total = eletrodosA + eletrodosC
    
    combined = list(zip(total, classes))
    random.shuffle(combined)

    total[:], classes[:] = zip(*combined)
    a = None

    for i in total:
        if len(i) != 256:
            a = total.index(i)
            total.pop(a)
            classes.pop(a)



    return [np.asarray(total), np.asarray(classes)]

def training(data, name):
    classifier = Sequential()

    classifier.add(Dense(units = 328, activation = 'relu', input_dim = 256))
    classifier.add(Dense(units = 278, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    classifier.fit(data[0], data[1], batch_size = 890, nb_epoch = 15)


    classifier.save(name)

def test_model(test, name_model):

    model = load_model(name_model)

    prev = model.predict(test[0])

    prev = (prev > 0.50)
    print(accuracy_score(prev, test[1]))
    matrix = confusion_matrix(prev, test[1])
    
if __name__ == '__main__':
    name = 'model.h5'
    train = load_bases2('large_train')
    test = load_bases2('large_test')

    train_processed = pre_pros(train)
    test_processed = pre_pros(test)


    training(train_processed, name)

    test_model(test_processed, name)
    
    



