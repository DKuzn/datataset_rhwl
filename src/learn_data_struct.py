import numpy as np
from PIL import Image
from dataclasses import dataclass
import os

@dataclass()
class LearnData:
    letter: str = None
    data: np.ndarray = None
    folder: str = None


class DataSet:
    def __init__(self, batch_size=10):
        '''
        :param batch_size: размер батча
        '''
        self.data_set = []
        #if batch_size > 0 and batch_size <= 63:
        self.batch_size = int(batch_size)
        #else:
            #raise ValueError('Batch size is too big.')
        self.letter_batch = self.__get_batch_letter()




    def __getitem__(self, key):
        '''
        :param key: Ключ возвращаемого значения
        :return: Значение запрошенное по ключу
        '''
        return self.data_set[key]

    def __add_data(self, path):
        '''
        :param path: Путь к изображению
        :return: None
        '''
        data = Image.open(path)
        letter = os.path.splitext(os.path.basename(path))[0]
        folder = str(path)
        np_data = np.array(data)
        learn_data = LearnData(letter, np_data, folder)
        self.data_set.append(learn_data)

    def __get_batch_letter(self):
        lb = []
        i = 0
        while i < self.batch_size:
            num = np.random.randint(0, 62+1)
            if num not in lb:
                lb.append(num)
                i += 1
        return lb

    def batch(self):
        for i in self.letter_batch:
            path = '../cut_img'
            path = os.path.join(path, str(np.random.randint(1, 8 + 1)))
            list_image = os.listdir(path)
            self.__add_data(os.path.join(path, list_image[i]))
        return self.data_set

a = DataSet(100)
batch = a.batch()
for i in batch:
    print(i.folder)