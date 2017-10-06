import io
import bson  # this is installed with the pymongo package
from skimage.io import imread, imsave  # or, whatever image library you prefer
import multiprocessing as mp  # will come in handy due to the size of the data
import os
import pandas as pd
from tqdm import tqdm

import params

base_dir = params.base_dir

bson_file = 'input/train.bson'
image_dir = 'train_images'
image_path = os.path.join(base_dir, image_dir)
bson_file_path = os.path.join(base_dir, bson_file)

if not os.path.exists(image_path):  # if it does not exists, we create it
    os.makedirs(image_path)

max_items = 7069896
NCORE = 8


def process(q, iolock):
    while True:
        d = q.get()
        if d is None:
            break
        product_id = d['_id']
        for e, pic in enumerate(d['imgs']):
            picture = imread(io.BytesIO(pic['picture']))
            imsave(os.path.join(image_path, str(product_id) + "_" + str(e) + ".png"), picture)


if __name__ == '__main__':

    q = mp.Queue(maxsize=NCORE)
    iolock = mp.Lock()
    pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))

    product_ids = []
    category_ids = []
    n_pics = []

    # process the file

    data = bson.decode_file_iter(open(bson_file_path, 'rb'))
    for c, d in tqdm(enumerate(data), total=max_items):
        if (c + 1) > max_items:
            break
        product_id = d['_id']
        category_id = d['category_id']
        pics = d['imgs']
        product_ids.append(product_id)
        category_ids.append(category_id)
        n_pics.append(len(pics))
        q.put(d)  # blocks until q below its max size

    # tell workers we're done

    for _ in range(NCORE):
        q.put(None)
    pool.close()
    pool.join()

    print('Images saved at %s' % image_path)

    # Match product_ids with their category_ids
    prod_to_category = pd.DataFrame(data={'product_id': product_ids, 'category_id': category_ids, 'n_pics': n_pics},
                                    columns=['product_id', 'category_id', 'n_pics'])
    prod_to_category.to_csv(path_or_buf="csv_files/prod_to_category.csv", sep=',', index=False)
