# Kaggle Cdiscount's Image Classification Challenge solution with Keras
My code for Cdiscount's Image Classification Challenge. Tested on a subset of 10k samples from the ~7m on the full dataset. I will add results on the full dataset once I'm able.

---

## Requirements
* Keras 2.0 w/ TF backend
* sklearn
* skimage
* tqdm
* h5py
* imgaug

---

## Usage
1) In `params.py` set `base_dir` to your working directory. You can also set the model to use and training parameters.
2) Place *train.bson* and *test.bson* in *{work_dir}/input*.
3) Run `read_data_train.py` and `read_data_test.py` to read and unpack train data and test data respectively.
4) Run `train.py` to train the model and `test_submit.py` to predict on test data and generate submission file.
