import numpy as np
from PIL import Image
from dataclasses import dataclass
import os

@dataclass()
class LearnData:
    letter: str = None
    data: np.ndarray = None


class DataSet:
    def __init__(self, batch_size=10):
        self.data_set = []
        if batch_size > 0 and batch_size <= 63:
            self.batch_size = int(batch_size)
        else:
            raise ValueError('Batch size is too big.')
        self.letter_batch = self.__get_batch_letter()

    def __str__(self):
        return "[\n\t" + 'letter:' + str(i.letter) + '\n' + '\t' + 'img:\n'  + str(i.data) + '\n' for i in self.data_set


    def __getitem__(self, item):
        return self.data_set[item]

    def __add_data(self, path):
        data = Image.open(path)
        letter = os.path.splitext(os.path.basename(path))[0]
        np_data = np.array(data)
        learn_data = LearnData(letter, np_data)
        self.data_set.append(learn_data)

    def show_data(self):  #метод под вопросом
        return self.data_set

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
        path = '../cut_img'
        sub_folder = np.random.randint(1,8+1)
        path = os.path.join(path, str(sub_folder))
        list_image = os.listdir(path)
        for i in self.letter_batch:
            self.__add_data(os.path.join(path, list_image[i]))
        return self.data_set

a = DataSet(2)
a.batch()
print(a)