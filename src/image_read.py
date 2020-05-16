import random
import os
from PIL import Image, ImageDraw #Подключим необходимые библиотеки.


def cut_image():
    for k in range(1, 9):
        path = '../sources_raw/'
        mode = image.mode
        size_old_image = image.size
        size_new_img: tuple = (size_old_image[0] - 450, size_old_image[1] - 809)
        color = 89
        test = Image.new(mode, size_new_img, color)
        image = Image.open(path + str(k) + ".jpg")  # Открываем изображение.
        mode = image.mode
        size_old_image = image.size
        size_new_img: tuple = (size_old_image[0] - 450, size_old_image[1] - 809)
        for i in range(size_old_image[0]):
            for j in range(size_old_image[1]):
                if 178 < i < size_new_img[0] + 178 and 88 < j < size_new_img[1] + 88:
                    test.putpixel((i-178, j-88), image.getpixel((i,j)))
        test.save(path + str(k) + 'new.jpg')


def rename():
    alphabet = 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщъыьЭэЮюЯя'
    for k in range(1, 9):
        dir = os.path.join('../cut_img/', str(k))
        jpg = os.listdir(dir)
        for i in jpg:
            index = int(os.path.splitext(i)[0])
            os.rename(os.path.join(dir, i), os.path.join(dir, alphabet[index] + '.jpg'))
    return jpg


print(rename())
