import json

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep

from glob import glob
from os import chdir, getcwdb
from os.path import join
import sys
import uuid

destino = r'C:\qq\06_automacao\captcha\extracao'
chdir(destino)
destino = join(destino, 'imagens\{}.png')

with open('dados.json') as f:
    dados = json.load(f)
    print(dados)

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--kiosk-printing')

driver = webdriver.Chrome(ChromeDriverManager().install(),
                          options=chrome_options)

qtd_max = 2
qtd = 0

driver.maximize_window()  
link = dados['link']


while True:
    
    if qtd >= qtd_max:
        break
    
    print('salvando', len(glob(destino.format('*'))))

    driver.get(link)
    
    sleep(5)
    
    e = driver.find_element_by_xpath(dados['xpath'])
    driver.execute_script("arguments[0].scrollIntoView();", e)
    
    uuid_ctrl = str(uuid.uuid4())
    destino_aj = destino.format(uuid_ctrl)
    
    with open(destino_aj, 'wb') as file:
        file.write(e.screenshot_as_png)
    
    sleep(10)
    
    qtd += 1
    
driver.quit()
