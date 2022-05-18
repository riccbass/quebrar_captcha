# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:52:21 2022

@author: ricar
"""




import sys
sys.path.insert(0, r'C:\qq\06_automacao\producao_captcha')

from keras.models import load_model
from helpers import resize_to_fit

from glob import glob
import numpy as np
import cv2 
import pickle
from shutil import copy2

from os.path import join, basename


with open(r'C:\qq\06_automacao\producao_captcha\modelo\lb.dat', 'rb') as f:
    lb = pickle.load(f)
       
modelo = load_model(r'C:\qq\06_automacao\producao_captcha\modelo\modelo_completo.dat')

dados = []
imagens = glob(r'C:\qq\06_automacao\modelo_captcha\caracteres\*.png')

for arquivo in imagens:
    
    print(arquivo)
    
    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    #padroninar 20 por 20
    
    imagem = resize_to_fit(imagem, 20, 20)
    
    #adicionar dimensão
    imagem = np.expand_dims(imagem, axis=2)    
    imagem = np.expand_dims(imagem, axis=0)    
       
    #adicionar a lista de dados
    
    letra_prevista = modelo.predict(imagem)
    letra_prevista = lb.inverse_transform(letra_prevista)[0]
    
    destino = join(r'C:\qq\06_automacao\modelo_captcha\caracteres_categorizados', letra_prevista)
    destino = join(destino, 'auto_' + basename(arquivo))
    
    copy2(arquivo, destino)
        
#refazer modelo