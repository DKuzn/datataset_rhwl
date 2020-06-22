from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import tensorflow as tf
import random as rd

AUTOTUNE = tf.data.experimental.AUTOTUNE
ALPHABET = 'А а Б б В в Г г Д д Е е Ё ё Ж ж З з И и Й й К к Л л М м \
            Н н О о П п Р р С с Т т У у Ф ф Х х Ц ц Ч ч Ш ш Щ щ ъ ы ь Э э Ю ю Я я'.split()

path = '../cut_img_old/'

all_img_paths = []
for i in range(1, 9):
    for j in range(len(ALPHABET)):
        all_img_paths.append(path + str(i) + '/' + ALPHABET[j] + '.jpg')

rd.shuffle(all_img_paths)

image_count = len(all_img_paths)

label_to_index = dict((name, index) for index, name in enumerate(ALPHABET))

all_img_labels = [label_to_index[pathlib.Path(path).name[0]]
                  for path in all_img_paths]

img_path = all_img_paths[0]

img_raw = tf.io.read_file(img_path)

img_tensor = tf.image.decode_image(img_raw)

img_final = tf.image.resize(img_tensor, [128, 128])
img_final = img_final / 255.0


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_img_paths)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_img_labels, tf.int64))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
