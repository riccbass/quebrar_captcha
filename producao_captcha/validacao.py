# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:18:31 2022

@author: ricar
"""

import sys

sys.path.insert(0, r'C:\qq\06_automacao\captcha\producao_captcha')

from ler_captcha_imagem import ler_captcha

from keras.models import load_model
import pickle

from datetime import datetime

from os import remove

from os.path import join, basename

import pandas as pd
import xml.etree.ElementTree as ET

from glob import glob

with open(r'C:\qq\06_automacao\captcha\producao_captcha\modelo\lb.dat', 'rb') as f:
    lb = pickle.load(f)
       
modelo = load_model(r'C:\qq\06_automacao\captcha\producao_captcha\modelo\modelo_completo.dat')

lst_anal = []
imagens = []
xmls = glob(r'C:\qq\06_automacao\captcha\extracao\resumo\*.xml')

for xml in xmls:
    
    tree = ET.parse(xml)
    root = tree.getroot()
    
    root.iter()
    
    filename = None
 
    for child in root.iter():
        
        if child.tag == 'path':
            
            filename = child.text
            break
        
    letras = []
        
    for child in root.iter():
                         
        if child.tag == 'object':
                
            caracter = None
            x = None
            
            for grandchild in child.iter():
                
                if grandchild.tag == 'name':
                    
                    caracter = grandchild.text
                    
                if grandchild.tag == 'bndbox':
                    
                    for grandgrandchild in grandchild.iter():
                        
                        if grandgrandchild.tag == 'xmin':
                    
                            x = grandgrandchild.text
                            
                            letras.append({'x':x,
                                           'caracter':caracter})
                            
    letras.sort(key = lambda x : int(x['x']))
    
    texto = [i['caracter'] for i in letras]
    texto = ''.join(texto)
    
    filename_aj = join(r'C:\qq\06_automacao\captcha\extracao\imagens' ,basename(filename))
        
    texto_captcha = ler_captcha(filename_aj, modelo, lb)   
    
    imagens.append(filename)
    
    teste_total = texto == texto_captcha

    lst_anal.append({'texto':texto,
                     'texto_captcha':texto_captcha,
                     'filename':filename,
                     'teste_total':teste_total})    

df_agg = pd.DataFrame(lst_anal)

print(df_agg['teste_total'].value_counts() / len(df_agg))

anomesdia = datetime.now().strftime('%Y%m%d')

df_agg.to_excel(fr'C:\qq\06_automacao\captcha\producao_captcha\teste_validacao\teste_validacao_{anomesdia}.xlsx',
                index = False,
                sheet_name = 'VALIDACAO')

'''
imagens_tot = glob(r'C:\qq\06_automacao\extracao\imagens\*.png')

for imagem in imagens_tot:
    
    if imagem not in imagens:
        print('n existe')
        remove(imagem)
'''