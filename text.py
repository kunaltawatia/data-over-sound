import numpy as np


def read_txt(file_name: str):
    with open(file_name, 'r') as fopen:
        txt = fopen.read()
        return txt


def write_txt(file_name: str, txt: str):
    with open(file_name, 'w') as fopen:
        fopen.write(txt)


def txt_to_data(txt: str):
    data = np.zeros(len(txt) * 2, dtype=np.int16)
    for idx, c in enumerate(txt):
        code = ord(c)
        data[2*idx] = code // 16
        data[2*idx + 1] = code % 16
    return data


def data_to_txt(data: list):
    txt = ''
    for idx in range(0, len(data), 2):
        code = data[idx] * 16
        code += data[idx + 1]
        txt += chr(int(code))
    return txt
