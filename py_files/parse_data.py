import pandas as pd
import os
import mne
import numpy as np

# Функция, которая загружает всю информацию об артефактах
# Artifact data downloading
def read_markers_from_dir(dir):
    # dir - str (marker files directories)
    paths = os.listdir(dir)

    res = []
    for path in paths:
        # Проверка на отстутствие расширения (в одной директории находятся файлы .edf / check if file doesn't have sufix
        # и файлы с маркировкой артефактов, у которых отстствует расширение)
        if path[-4] == '.':
            continue
        file = open(dir + path, 'r')
        freq = float(file.readline().split(':')[2][:-3])# Чтение первой строки, содержащей частоту снятия показаний / reading frequency value in first raw (3.9 ms)
        file.close()
        info = pd.read_csv(dir + path, header=1)[1:]
        # Перевод в секунды значений позиции и длинны / transformation  from seconds to values (position and length)
        info[' Position'] =  info[' Position']*freq/1000
        info[' Length'] = info[' Length'] * freq / 1000
        res.append({'name': path.split('_')[0], 'content': info})
    # returns dict (associative array of artifacts)
    return res