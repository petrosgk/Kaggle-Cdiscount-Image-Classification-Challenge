import os
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from skimage.io import imread, imsave
from skimage.transform import resize
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop

from models import Models
import params

height = params.height
width = params.width
batch_size = params.batch_size
max_epochs = params.max_epochs

base_dir = params.base_dir
image_dir = 'train_images'

csv_file = 'csv_files/prod_to_category.csv'

df = pd.read_csv(csv_file)
product_ids, category_ids, n_pics = df['product_id'], df['category_id'], df['n_pics']
product_ids = np.array(product_ids, np.uint32)
category_ids = np.array(category_ids, np.uint32)
n_pics = np.array(n_pics, np.uint8)

# Convert category_id's to categorical labels
unique_categories = np.unique(category_ids)
classes = len(unique_categories)
labels = np.arange(classes)
categorical_labels = to_categorical(labels)
categories_to_labels = defaultdict()
# Match categories with their respective categorical labels
for i in range(classes):
    categories_to_labels[unique_categories[i]] = i
category_to_label = pd.DataFrame(list(categories_to_labels.items()), columns=['category_id', 'label_index'])
category_to_label.to_csv('csv_files/category_to_label.csv', index=False)

product_ids_train, product_ids_test, \
category_ids_train, category_ids_test, \
n_pics_train, n_pics_test = train_test_split(product_ids,
                                             category_ids,
                                             n_pics,
                                             test_size=0.1,
                                             random_state=42,
                                             stratify=category_ids)

print('Train on {} product_ids, validate on {} product_ids'.format(len(product_ids_train), len(product_ids_test)))

# Build train and validation samples arrays
product_ids_pics_train_arr = []
category_ids_train_arr = []
print('Creating training data:')
for prod in tqdm(range(len(product_ids_train))):
    for pic in range(n_pics_train[prod]):
        product_ids_pics_train_arr.append(str(product_ids_train[prod]) + '_' + str(pic))
        category_ids_train_arr.append(category_ids_train[prod])
product_ids_pics_test_arr = []
category_ids_test_arr = []
print('Creating validation data:')
for prod in tqdm(range(len(product_ids_test))):
    for pic in range(n_pics_test[prod]):
        product_ids_pics_test_arr.append(str(product_ids_test[prod]) + '_' + str(pic))
        category_ids_test_arr.append(category_ids_test[prod])
train_samples = len(product_ids_pics_train_arr)
test_samples = len(product_ids_pics_test_arr)


def augmenter(image, u):
    def sometimes(p, aug):
        return iaa.Sometimes(p, aug)

    seq = iaa.Sequential(
        [iaa.Fliplr(u),  # Horizontal flip
         sometimes(u, iaa.Affine(
             rotate=(-180, 180),  # Rotate by -180 to +180 degrees
             scale=(0.8, 1.2),  # Scale by 80% to 120%
             translate_percent=(0.0, 0.1)
         ))]
    )

    return seq.augment_image(image)


def train_generator():
    while True:
        for start in range(0, train_samples, batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, train_samples)
            product_ids_pics_train_arr_batch = product_ids_pics_train_arr[start:end]
            category_ids_train_arr_batch = category_ids_train_arr[start:end]
            for sample in range(len(product_ids_pics_train_arr_batch)):
                img = os.path.join(base_dir, image_dir, product_ids_pics_train_arr_batch[sample] + '.png')
                img = imread(img)
                assert img is not None, 'Failed to read image'
                img = img[:, :, :3]
                img = resize(img, (height, width), preserve_range=True)
                # img = augmenter(img, u=0.5)
                # imsave(os.path.join(base_dir, 'augs',  product_imgs_train_batch[sample] + '.png'), img / 255)
                x_batch.append(img)
                label_index = categories_to_labels[category_ids_train_arr_batch[sample]]
                label = categorical_labels[label_index]
                y_batch.append(label)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, test_samples, batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, test_samples)
            product_ids_pics_test_arr_batch = product_ids_pics_test_arr[start:end]
            category_ids_test_arr_batch = category_ids_test_arr[start:end]
            for sample in range(len(product_ids_pics_test_arr_batch)):
                img = os.path.join(base_dir, image_dir, product_ids_pics_test_arr_batch[sample] + '.png')
                img = imread(img)
                assert img is not None, 'Failed to read image'
                img = img[:, :, :3]
                img = resize(img, (height, width), preserve_range=True)
                x_batch.append(img)
                label_index = categories_to_labels[category_ids_test_arr_batch[sample]]
                label = categorical_labels[label_index]
                y_batch.append(label)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32)
            yield x_batch, y_batch


models = Models(input_shape=(height, width, 3), classes=classes)
if params.base_model == 'vgg16':
    models.vgg16()
elif params.base_model == 'vgg19':
    models.vgg19()
elif params.base_model == 'resnet50':
    models.resnet50()
elif params.base_model == 'inceptionV3':
    models.inceptionV3()
else:
    print('Uknown base model')
    raise SystemExit

callbacks = [ModelCheckpoint(filepath='weights/best_weights_' + params.base_model + '.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             ReduceLROnPlateau(factor=0.5,
                               patience=2,
                               verbose=1,
                               epsilon=1e-4),
             EarlyStopping(min_delta=1e-4,
                           patience=4,
                           verbose=1)]

models.compile(optimizer=RMSprop(lr=1e-5))
model = models.get_model()
model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(train_samples) / float(batch_size)),
                    epochs=max_epochs,
                    verbose=1,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(test_samples) / float(batch_size)),
                    callbacks=callbacks)
