import numpy as np

def read_pgm(file_name: str):
    with open(file_name, 'r') as fopen:
        typ = fopen.readline().strip()
        w, h = map(int, fopen.readline().strip().split())
        mx_value = int(fopen.readline().strip())
        img = np.zeros((h, w))
        for row in range(h):
            img[row] = fopen.readline().strip().split()
        return img


def write_pgm(file_name: str, image: np.ndarray):
    with open(file_name, 'w') as fopen:
        print('P2', file=fopen)
        h, w = image.shape
        print(w, h, file=fopen)
        print('15', file=fopen)
        for row in image:
            for val in row:
                print(int(val), end=' ', file=fopen)
            print(file=fopen)


def image_to_data(image: np.ndarray):
    return list(map(int, image.flatten()))


def data_to_image(data: list):
    w, h = 24, 7
    image = np.array(data).reshape(h, w)
    return image
