# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:33:27 2022

@author: ricar
"""

import cv2

import tensorflow as tf
import numpy as np
import pickle

from glob import glob
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import TensorBoard

from os.path import join

import sys
sys.path.insert(0, r'C:\qq\06_automacao\captcha\producao_captcha')

from helpers import resize_to_fit

import hashlib

pasta_base_imagens = r'C:\qq\06_automacao\captcha\modelo_captcha\caracteres_categorizados'
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
    
dados = []
rotulos = []
    
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
    
X = np.array(dados, dtype="float") / 255
y = np.array(rotulos)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, y, test_size=0.0001)

lb = LabelBinarizer().fit(Y_train)
y_lb = lb.transform(Y_train)

#salvar o label binalzier

with open(r'C:\qq\06_automacao\captcha\producao_captcha\modelo\lb.dat', 'wb') as ap:
    pickle.dump(lb, ap)
   
    
layer_sizes = [100, 250, 500, 750]
kernel_sizes = [4, 5]
conv_layers = [1, 2, 3]
dense_layers = [0, 1, 2]
lrs = [0.001, 0.01]

'''
melhor é:
    
2 conv, 1 dense, 500 lz e 4 ks
'''

layer_sizes = [500]
kernel_sizes = [4]
conv_layers = [2]
dense_layers = [1]
lrs = [0.001]
batches_sizes = [50]

for batch_size in batches_sizes:
    
    for lr in lrs:
        
        for dense_layer in dense_layers:
            
            for conv_layer in conv_layers:
                    
                for layer_size in layer_sizes:
                        
                    for kernel_size in kernel_sizes:
                        
                        NAME = f'{batch_size}_bs_{conv_layer}_conv_{lr}_lr_{dense_layer}_dense_20_layer_size_conv_{layer_size}_layer_size_{kernel_size}_ks'    
                        tensorboard = TensorBoard(log_dir=fr"C:\qq\06_automacao\captcha\logs\{NAME}")
                        
                        modelo = Sequential()
                        
                        modelo.add(Conv2D(20, 
                                          (kernel_size, kernel_size), 
                                          padding="same", 
                                          input_shape=(20, 20, 1), 
                                          activation="relu"))
                        modelo.add(Activation('relu'))           
                        modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                                   
                        for l in range(conv_layer - 1):
                        
                            modelo.add(Conv2D(50, (kernel_size, kernel_size), padding="same"))
                            modelo.add(Activation('relu'))  
                            modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                            
                        modelo.add(Flatten())
            
                        for l in range(dense_layer):
                                
                            modelo.add(Dense(layer_size))
                            modelo.add(Activation('relu'))
                            modelo.add(Dropout(0.2))
                            
                        
                        #saída
                        modelo.add(Dense(len(set(rotulos)), activation = "softmax"))
                        
                        #compilar
                        
                        opt = tf.keras.optimizers.Adam(learning_rate=lr)
                              
                        modelo.compile(loss="categorical_crossentropy", 
                                       optimizer=opt,
                                       metrics=["accuracy"])
                        
                        modelo.fit(X_train, 
                                   y_lb, 
                                   batch_size=batch_size,
                                   epochs = 9,
                                   validation_split=0.1,
                                   callbacks=[tensorboard])

#0.9677

#modelo.save(r'C:\qq\06_automacao\producao_captcha\modelo\modelo_completo.dat')
   
