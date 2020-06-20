import numpy as np
from PIL import Image
from dataclasses import dataclass
import os

@dataclass()
class LearnData:
    letter: str = None
    data: np.ndarray = None


class DataSet:
    def __init__(self):
        self.__path_to_trained_data = '../cut_img/trained_data'
        self.__path_to_test_data = '../cut_img/test_data'
        self.__data_trained = self.__get_data(self.__path_to_trained_data)
        self.__data_test = self.__get_data(self.__path_to_test_data)
        self.__all_data = self.__get_all_data()
        self.__all_classes = self.__get_classes()

    def __getitem__(self, key):
        '''
        :param key: Ключ возвращаемого значения
        :return: Значение запрошенное по ключу
        '''
        return self.data_set[key]

    def __add_data(self, path, list_append):
        data = Image.open(path)
        letter = os.path.splitext(os.path.basename(path))[0]
        np_data = np.array(data)
        learn_data = LearnData(letter, np_data)
        list_append.append(learn_data)


    def __get_data(self, path):
        list_directory = os.listdir(path)
        data = []
        for i in list_directory:
            path_to_img = os.path.join(path, i)
            list_images = os.listdir(path_to_img)
            for j in list_images:
                self.__add_data(path=os.path.join(path_to_img, j), list_append=data)
        return data

    def __get_all_data(self):
        data = []
        for i in self.__data_trained:
            data.append(i)
        for i in self.__data_test:
            data.append(i)
        return data

    def __get_classes(self):
        data = []
        list_img = os.listdir('../cut_img/test_data/7')
        for i in list_img:
            letter = os.path.basename(i)[0]
            data.append(letter)
        return data

    def get_trained_data(self):
        return self.__data_trained

    def get_test_data(self):
        return self.__data_test

    def get_all_data(self):
        return self.__all_data

    def get_all_classes(self):
        return self.__all_classes
