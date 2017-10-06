import os
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize
from keras.utils.np_utils import to_categorical

from models import Models
import params

height = params.height
width = params.width
batch_size = params.batch_size

base_dir = params.base_dir
image_dir = 'test_images'

csv_file_test = 'csv_files/prod_to_pic.csv'
csv_file_labels = 'csv_files/category_to_label.csv'

df_test = pd.read_csv(csv_file_test)
product_ids = df_test['product_id']
n_pics = df_test['n_pics']
product_ids = np.array(product_ids, np.uint32)
n_pics = np.array(n_pics, np.uint8)

# Retrieve categories and their corresponding categorical labels
df_labels = pd.read_csv(csv_file_labels)
category_ids, label_indexes = df_labels['category_id'], df_labels['label_index']
category_ids = np.array(category_ids, np.uint32)
label_indexes = np.array(label_indexes, np.uint32)

# Convert categorical labels to category_ids
classes = len(category_ids)
labels = np.arange(classes)
categorical_labels = to_categorical(labels).astype(np.uint8)
labels_to_categories = defaultdict()
for i in range(classes):
    labels_to_categories[label_indexes[i]] = category_ids[i]

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

models.load_weights('weights/best_weights_' + params.base_model + '.hdf5')
model = models.get_model()

test_samples = len(product_ids)
pred_category_ids = []
for sample in tqdm(range(0, test_samples)):
    x_batch = []
    for pic in range(n_pics[sample]):
        img = os.path.join(base_dir, image_dir, str(product_ids[sample]) + '_' + str(pic) + '.png')
        img = imread(img)
        img = img[:, :, :3]
        img = resize(img, (height, width), preserve_range=True)
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    # Get the average prediction over all images of this product
    pred = np.average(preds, axis=0)
    # Predicted label should have 1 as the predicted class and the rest 0
    pred_label = np.zeros_like(pred, np.uint8)
    # Find class with highest probability and set it to 1
    max_p_ind = np.argmax(pred)
    pred_label[max_p_ind] = 1
    # Match predicted label with its corresponding category_id
    for label_idx in range(categorical_labels.shape[0]):
        if np.array_equal(categorical_labels[label_idx], pred_label):
            pred_category_ids.append(labels_to_categories[label_idx])

# Match product_ids with their predicted category_ids
submission = pd.DataFrame({'_id': product_ids, 'category_id': pred_category_ids},
                          columns=['_id', 'category_id'])
submission.to_csv(path_or_buf='csv_files/submission.csv.gz', sep=',', index=False, compression='gzip')

