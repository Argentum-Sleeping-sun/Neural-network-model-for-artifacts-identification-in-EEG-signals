import numpy as np
import collections

# Функция, которая определяет максимальную длинну артефакта
# Finding max artifact length
def max_artifact_length(markers):
    # markers - dict (associative array of artifacts)
    res = 0
    space = -1
    for marker in markers:
        mx = max(marker['content'][' Length'])
        if mx > res:
            res = mx
            space = marker['name']
    print('Max length found at name: ', space)
    # returns int (max artifact length)
    return res

# Функция, которая находит максимальную длинну определённого типа артефакта
# Finding max artifact length for specific type
def max_type_length(markers, type):
    # markers - dict (associative array of artifacts)
    # type - str (specific type of artifact)
    res = 0
    space = -1
    for marker in markers:
        mx = 0
        for i, it in enumerate(marker['content'][' Length']):
            if marker['content'][' Description'][i+1] == type and it > mx:
                mx = it
        if mx > res:
            res = mx
            space = marker['name']
    print('Max length found at name: ', space)
    # returns int (max artifact length in specific type)
    return res

# Функция, которая считает количество артефактов для каждого типа
# Finding the number of artifacts for every type
def channels_stats(markers):
    # markers - dict (associative array of artifacts)
    res = collections.Counter()
    for marker in markers:
        res.update(marker['content'][' Channel'])
    # returns dict (number of artifacts for every type)
    return res

# Функция, которая ищет все расстояния между артефактами
# Finding lengths between artifacts (approximation - generalization value)
def normal_state_lengths(markers, approximation = 1):
    # markers - dict (associative array of artifacts)
    # approximation - int (approximation value)
    res = collections.Counter()
    acum = 0
    for marker in markers:
        prev_point = 0
        for i, pos in enumerate(marker['content'][' Position']):
            distance = pos - prev_point
            acum += 1
            
            res.update({int(distance/approximation): 1})
            prev_point = pos + marker['content'][' Length'][i+1]
    # returns dict (number of spaces between artifacts for every distance value)
    return res
