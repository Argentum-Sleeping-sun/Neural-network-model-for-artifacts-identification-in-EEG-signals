from functools import partial

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, \
    LSTM, BatchNormalization, MaxPool1D, Dropout, Conv2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import categorical_accuracy
import pyedflib
from scipy.signal import spectrogram
import random


# Функция, которая применяет быстрое преобразование Фурье к сигналу
# Function that applies fast Fourier transform (fft) to signal
def filter_signal(signal, threshold=1e3):
    # signal - vector of floats (signal values)
    # threshold - float
    fourier = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0
    # returns fft of signal (numpy array)
    return np.fft.irfft(fourier)
    
# Преобразование каналов с помощью преобразования Фурье, перевод в вектор
# Functions that applies fft to signal channels and formes single vector
def vectorize_edf_frame(point, frame_size, chns, edf_info, freq):
    # point - int (start position in edf file)
    # frame_size - int (size of time window (in seconds))
    # chns - channels to record
    # edf_info - edf file
    # freq - sampling frequency
    global acum
    train_x = []
    for it2 in chns:
        train_x.extend(filter_signal(edf_info.readSignal(it2, point * freq[it2], frame_size * freq[it2])))
    # returns vectorized frames of signals
    return train_x

# Формирование монтажей (разницы между каналами), преобразование Фурье и приведение к вектору
# Function that forms montage, applies fft to it and forms a single vector
def vectorize_montage_frame(point, frame_size, chns, edf_info, freq):
    # point - int (start position in edf file)
    # frame_size - int (size of time window (in seconds))
    # chns - channels to record
    # edf_info - edf file
    # freq - sampling frequency
    global acum
    train_x = []
    for it2 in chns:
        train_x.extend(filter_signal(np.subtract(edf_info.readSignal(it2[0], point * freq[it2[0]], frame_size * freq[it2[0]]),
                                                 edf_info.readSignal(it2[1], point * freq[it2[1]], frame_size * freq[it2[1]]))))
    # returns vectorized frames of montages
    return train_x

#Генерирование новых данных на основе имеющихся
# Data augmentation
def blink_augmentation(position, frame_size, length, value = 3):
    # position - int (start position in edf file)
    # frame_size - int (size of time window (in seconds))
    # length - int (length of artifact)
    # value - int (number of positions to generate)
    fat_to_left = (frame_size - length) * random.random()
    positions = np.zeros(value)
    for i in range(value):
        positions[i] = position - fat_to_left/(i+1)
    # returns numpy array of positions for sampling
    return positions

# Функция, которая по данным с маркированными артефактами, формирует выборки для нейронной сети
# Function for sample generating
def prepare_data(markers, edf_dir, chnls=[0], frame_size=2, normal_state_percentage=0.3):
    # markers - dict (associative array of artifacts)
    # edf_dir - str (edf files directory)
    # chnls - list (channel indexes to use in samples)
    # frame_size - int (size of time window (in seconds) for every sample)
    # normal_state_percentage - ratio of notmal state samples in all samples
    train_x, train_y = [],[]
    normal_state_x = []
    for marker in markers:
        name = marker['name']
        edf_info = pyedflib.EdfReader(edf_dir + name + '_Connectivity.edf')# Чтение EDF файла / EDF reading
        freq = edf_info.getSampleFrequencies()# Количество снятых показаний в секунду / Sampling frequency (seconds)
        prev_point = 0
        for i, it in enumerate(marker['content'][' Description']):
            point = 0
            description = it # Тип артефакта / Artifact type
            position = marker['content'][' Position'][i+1]# Позиция артефакта (в секундах) / Artifact position (in seconds)
            length = marker['content'][' Length'][i+1]# Длинна артефакта (в секундах) / Artifact length (in seconds)
            
            # Для артефакта моргания, выбирается промежуток времени, который полностью его охватывает
            # For a blinking artifact, a period of time is chosen that completely covers it
            if description == ' Blink' and marker['content'][' Channel'][i+1] == ' Fp1':
                # Расположение артефакта в выборке, находится случайным образом
                # Artifact is positioned randomly in every sample
                for point in blink_augmentation(position, frame_size, length):
                    # Если позиция для считывания меньше нуля, выборка игнорируется / Ignoring negative positions
                    if point > 0:
                        train_x.append(vectorize_edf_frame(point, frame_size, chnls, edf_info, freq))
                        train_y.append([1, 0, 0])
            # Для всех остальных артефактов, берутся выборки по всей их длине 
            # For all other artifacts, samples are taken along its entire length (depends on artifact length)
            if description == ' Userdefined':
                point = position
                r_border = position + length
                while point < r_border:
                    train_x.append(vectorize_edf_frame(point, frame_size, chnls, edf_info, freq))
                    train_y.append([0, 1, 0])
                    shift = frame_size / 3
                    point += shift
            # Если расстояние между артефактами больше, чем размер входа, то формируется выборка с типом "нормальное состояние"
            # 'Normal state' samples forming if distance between artifacts is bigger then frame size
            while point - prev_point > frame_size:
                normal_state_x.append(vectorize_edf_frame(prev_point, frame_size, chnls, edf_info, freq))
                prev_point += frame_size

            prev_point = position + length
    # Опредение кол-ва артефактов с нормальным состоянием в выборке
    # Defining noraml state number based on normal state ratio in all samples
    normal_state_number = min(len(normal_state_x), int(normal_state_percentage * len(train_x)/(1 - normal_state_percentage)))
    for it in normal_state_x[:normal_state_number]:
        train_x.append(it)
        train_y.append([0, 0, 1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # returns input and output samples (numpy arrays)
    return train_x, train_y