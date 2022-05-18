# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:33:27 2022

@author: ricar
"""

import cv2

import numpy as np
import pickle

from glob import glob
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense

from os.path import join

import sys
sys.path.insert(0, r'C:\qq\06_automacao\producao_captcha')

from helpers import resize_to_fit

import hashlib

dados = []
rotulos = []

pasta_base_imagens = r'C:\qq\06_automacao\modelo_captcha\caracteres_categorizados'
imagens = glob(join(pasta_base_imagens,'**\*.png'), recursive = True)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

imagens_aj = []
hashes = set()

for arquivo in imagens:
    
    md5_arquivo = md5(arquivo)
    
    if md5_arquivo in hashes:
        continue
    
    hashes = list(hashes)
    hashes.append(md5_arquivo)
    
    imagens_aj.append(arquivo)
    
for arquivo in imagens_aj:
    
    rotulo = arquivo.split('\\')[-2]
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    #padroninar 20 por 20
    
    imagem = resize_to_fit(imagem, 20, 20)
    
    #adicionar dimensão
    imagem = np.expand_dims(imagem, axis=2)    
    
    
    #adicionar a lista de dados
    
    rotulos.append(rotulo)
    dados.append(imagem)    
    
dados = np.array(dados, dtype="float") / 255
rotulos = np.array(rotulos)

(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)
    
#salvar o label binalzier

with open(r'C:\qq\06_automacao\producao_captcha\modelo\lb.dat', 'wb') as ap:
    pickle.dump(lb, ap)
    
modelo = Sequential()

modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


modelo.add(Conv2D(50, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

modelo.add(Flatten())
modelo.add(Dense(500, activation = "relu"))

#saída
modelo.add(Flatten())
modelo.add(Dense(len(set(rotulos)), activation = "softmax"))

#compilar

modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#treinar
modelo.fit(X_train, 
           Y_train, 
           validation_data=(X_test, Y_test), 
           batch_size = len(set(rotulos)),
           epochs=20, verbose=1)

modelo.save(r'C:\qq\06_automacao\producao_captcha\modelo\modelo_completo.dat')
   
