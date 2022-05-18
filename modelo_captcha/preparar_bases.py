# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:51:56 2022

@author: ricar
"""

#1) segmentar bases

# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:23:05 2022

@author: ricar
"""

import cv2
import numpy as np
from glob import glob
import pandas as pd

from os.path import join, basename, exists, splitext

origem = r'C:\qq\06_automacao\extracao\imagens'
destino = r'C:\qq\06_automacao\modelo_captcha\imagens_marcadas'
destino_caracter = r'C:\qq\06_automacao\modelo_captcha\caracteres'

def cor(n):
    
    if n == 0:
        return (255,0,0) #vermelho
    elif n == 1:
        return (0,0,255) #azul
    elif n == 2:
        return (255,255,0) #amarelo
    elif n == 3:
        return (128,0,128) #roxo
    else:
        return (24,252,0) # verde
    
    
def ler_imagem(arquivo):
    
    image = cv2.imread(arquivo)
    
    return image
    
def ler_contornos(image):
        
    blur = cv2.medianBlur(image, 7)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
    
    canny = cv2.Canny(thresh, 120, 255, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    dilate = cv2.dilate(opening, kernel, iterations=2)
    
    cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    return cnts

def conver_preto_e_branco(arquivo):
    
    imagem = cv2.imread(arquivo)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #imagem_cinza = clahe.apply(imagem_cinza)
     
    _, imagem_tratada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
    cv2.imwrite(arquivo, imagem_tratada)
    


def salvar_caracteres(arquivo, image, cnts, destino_aj):
        
    lst_retorno = []
    
    if exists(destino_aj):
        return lst_retorno
    
    for n, c in enumerate(cnts):
       
       arquivo_aj = splitext(basename(arquivo))[0]
       
       destino_caracter_aj = join(destino_caracter, arquivo_aj + '_' + str(n) + '.png')
       
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
       cv2.imwrite(destino_caracter_aj, image_char)
       conver_preto_e_branco(destino_caracter_aj)
       
       cv2.rectangle(image, (x, y), (x + w, y + h), cor(n), 2)
                 
       dict_item = {'n':n,
                    'arquivo':arquivo_aj,
                    'destino':destino_caracter_aj,
                    'area':area,
                    'x':x,
                    'y':y,
                    'w':w,
                    'h':h}
       
       lst_retorno.append(dict_item)
       
    cv2.imwrite(destino_aj, image)

    return lst_retorno
   
arquivos = glob(join(origem, '*.png'))
lst_retorno_total = []    

for arquivo in arquivos:

    destino_aj = join(destino, basename(arquivo))
    
    if exists(destino_aj):
        continue
    
    print(f'salvando {arquivo}')
    
    image = ler_imagem(arquivo)    
    cnts = ler_contornos(image)
    
    lst = salvar_caracteres(arquivo, image, cnts, destino_aj)
    
    lst_retorno_total += lst
    


df = pd.DataFrame(lst_retorno_total)

df['n'] = 1

df_agg = (

    df.groupby(['arquivo'], as_index = False).agg({'n':'count'})
    
)

df_agg = df_agg[df_agg['n'] >= 5]

