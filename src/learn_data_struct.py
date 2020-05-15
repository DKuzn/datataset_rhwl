import numpy as np
from PIL import Image


class LearnData:
    def __init__(self, letter, number, data):
        self.letter: str = letter
        self.number: int = number
        self.data: np.ndarray = data


class DataSet:
    def __init__(self):
        self.data_set = []

    def __str__(self):
        current = self.data_set
        out = "[" + str(current[0].data)
        for i in range(1, len(self.data_set)):
            current = self.data_set[i].data
            out += ",\n" + str(current)
        return out + "]"

    def add_data(self, path: str, letter: str, number: int):
        self.data = Image.open(path)
        self.np_data = np.array(self.data)
        self.learn_data = LearnData(letter, number, self.np_data)
        self.data_set.append(self.learn_data)

    def show_data(self):  #метод под вопросом
        return self.data_set


dataset = DataSet()
dataset.add_data("../cut_img/1/а.jpg", "а", 1)
dataset.add_data("../cut_img/1/А.jpg", "А", 2)
print(dataset)