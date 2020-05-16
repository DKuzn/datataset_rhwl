import numpy as np
from PIL import Image
from dataclasses import dataclass
import os

@dataclass()
class LearnData:
    letter: str = None
    data: np.ndarray = None


class DataSet:
    def __init__(self, batch_size):
        self.data_set = []
        self.batch_size = batch_size

    def __str__(self):
        out = "["
        for i in self.data_set:
            out += ",\n" + 'letter:' + str(i.letter) + '\n' + str(i.data)
        return out + "]"

    def __getitem__(self, item):
        return self.data_set[item]

    def add_data(self, path):
        data = Image.open(path)
        letter = os.path.splitext(os.path.basename(path))[0]
        np_data = np.array(data)
        learn_data = LearnData(letter, np_data)
        self.data_set.append(learn_data)

    def show_data(self):  #метод под вопросом
        return self.data_set

    def __get_batch_letter(self):
        return np.random.randint(0, 62+1)

    def batch(self):
        path = '../cut_img'
        sub_folder = np.random.randint(1,8+1)
        path = os.path.join(path, str(sub_folder))
        list_image = os.listdir(path)
        for i in range(self.batch_size):
            self.add_data(os.path.join(path, list_image[self.__get_batch_letter()]))
        return self.data_set


a = DataSet(10)
a.batch()
print(a)