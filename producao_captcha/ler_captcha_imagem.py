# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:54:51 2022

@author: ricar
"""

import cv2
from keras.models import load_model
import pickle
import numpy as np
import sys

sys.path.insert(0, r'C:\qq\06_automacao\producao_captcha')

from helpers import resize_to_fit


def ler_imagem(arquivo):
    
    imagem = cv2.imread(arquivo)
    
    return imagem

def ler_contornos(imagem):
        
    blur = cv2.medianBlur(imagem, 7)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
    
    canny = cv2.Canny(thresh, 120, 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    dilate = cv2.dilate(opening, kernel, iterations=2)
    
    cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    return cnts

def salvar_caracteres(image, cnts):
        
    lst_retorno = []
    
    
    for n, c in enumerate(cnts):
          
       area = cv2.contourArea(c)
       x,y,w,h = cv2.boundingRect(c)
       
       '''
       remover lombrga
       '''
       
       if y <= 5 and h >= 45:
           continue
       
       '''
       remover grudado na direita
       '''
       
       if x >= 138:
           continue
       
       
       image_char = image[y:y+h, x:x+w]   
       #cv2.rectangle(image, (x, y), (x + w, y + h), cor(n), 2)
       imagem_cinza = cv2.cvtColor(image_char, cv2.COLOR_RGB2GRAY)
       _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
       
       imagem_resize = resize_to_fit(imagem_tratada, 20, 20)
          
                 
       dict_item = {'n':n,
                    'area':area,
                    'imagem_tratada':imagem_resize,
                    'x':x,
                    'y':y,
                    'w':w,
                    'h':h}
       
       lst_retorno.append(dict_item)
       
    return lst_retorno

def ler_captcha(arquivo, modelo, lb):
    
    imagem = ler_imagem(arquivo) #lê a imagem
    cnts = ler_contornos(imagem)   
    retornos = salvar_caracteres(imagem, cnts)
    
    retornos = sorted(retornos, key = lambda x: x['x'])
    texto_captcha = []
    
    for retorno in retornos:
        
        imagem_modelo = retorno['imagem_tratada']
        
        imagem_modelo = np.expand_dims(imagem_modelo, axis=2)    
        imagem_modelo = np.expand_dims(imagem_modelo, axis=0)    
        
        letra_prevista = modelo.predict(imagem_modelo)
        letra_prevista = lb.inverse_transform(letra_prevista)[0]
        texto_captcha.append(letra_prevista)
     
    texto_captcha = ''.join(texto_captcha)
    
    return texto_captcha
  


    