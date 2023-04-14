# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Machine Learning and Data Science Imports
import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import numpy as np
import sklearn
from tqdm import tqdm

# for running on CoLab or within Kaggle
# from kaggle_datasets import KaggleDatasets

# Mostly Builtins
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import Levenshtein
import warnings
import requests
import hashlib
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import math
import time
import gzip
import ast
import sys
import io
import os
import gc
import re

# Visualization Imports (overkill)
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import plotly.graph_objects as go
from IPython.display import HTML
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.express as px
import tifffile as tif
import seaborn as sns
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
from matplotlib import animation, rc; rc('animation', html='jshtml')
import plotly
import PIL

import plotly.io as pio
print(pio.renderers)

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_it_all()


# %% [markdown]
# ### Helper Functions

# %%
def flatten_l_o_l(nested_list):
    """Flatten a list of lists into a single list.

    Args:
        nested_list (list): 
            – A list of lists (or iterables) to be flattened.

    Returns:
        list: A flattened list containing all items from the input list of lists.
    """
    return [item for sublist in nested_list for item in sublist]


def print_ln(symbol="-", line_len=110, newline_before=False, newline_after=False):
    """Print a horizontal line of a specified length and symbol.

    Args:
        symbol (str, optional): 
            – The symbol to use for the horizontal line
        line_len (int, optional): 
            – The length of the horizontal line in characters
        newline_before (bool, optional): 
            – Whether to print a newline character before the line
        newline_after (bool, optional): 
            – Whether to print a newline character after the line
    """
    if newline_before: print();
    print(symbol * line_len)
    if newline_after: print();
        
        
def read_json_file(file_path):
    """Read a JSON file and parse it into a Python object.

    Args:
        file_path (str): The path to the JSON file to read.

    Returns:
        dict: A dictionary object representing the JSON data.
        
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the specified file path does not contain valid JSON data.
    """
    try:
        # Open the file and load the JSON data into a Python object
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        # Raise an error if the file path does not exist
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        # Raise an error if the file does not contain valid JSON data
        raise ValueError(f"Invalid JSON data in file: {file_path}")
        
def get_sign_df(pq_path, invert_y=True):
    sign_df = pd.read_parquet(pq_path)
    
    # y value is inverted (Thanks @danielpeshkov)
    if invert_y: sign_df["y"] *= -1 
        
    return sign_df


ROWS_PER_FRAME = 543  # number of landmarks per frame
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


# %% [markdown]
# ### Load Data

# %%
# Define the path to the root data directory
DATA_DIR         = str(os.getcwd()) + r"\asl-signs"
EXTEND_TRAIN_DIR = "/asl-signs/gislr-extended-train-dataframe" 

LOAD_EXTENDED = True
if LOAD_EXTENDED and os.path.isfile(os.path.join(EXTEND_TRAIN_DIR, "extended_train.csv")):
    train_df = pd.read_csv(os.path.join(EXTEND_TRAIN_DIR, "extended_train.csv"))
else:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_df["path"] = DATA_DIR+"\\"+train_df["path"]
display(train_df)

print("\n\n... LOAD SIGN TO PREDICTION INDEX MAP FROM JSON FILE ...\n")
s2p_map = {k.lower():v for k,v in read_json_file(os.path.join(DATA_DIR, "sign_to_prediction_index_map.json")).items()}
p2s_map = {v:k for k,v in read_json_file(os.path.join(DATA_DIR, "sign_to_prediction_index_map.json")).items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)
print(s2p_map)

DEMO_ROW = 283
print(f"\n\n... DEMO SIGN/EVENT DATAFRAME FOR ROW {DEMO_ROW} - SIGN={train_df.iloc[DEMO_ROW]['sign']} ...\n")
demo_sign_df = get_sign_df(train_df.iloc[DEMO_ROW]["path"])
display(demo_sign_df)

# %% [markdown]
# ### Sample Training Data
# #### There is a lot of data here (some ~50gb) so we're going to start with doing EDA on just some of the data

# %%
# During interactive --> 0.001 (0.1%)
# Save and run-all   --> 1.000 (100%)

PCT_TO_EXAMINE = 0.001
if PCT_TO_EXAMINE < 1.0:
    subsample_train_df = train_df.sample(frac=PCT_TO_EXAMINE, random_state=42).reset_index(drop=True)
else:
    subsample_train_df = train_df.copy()

# Remove extra columns to show what we're doing
subsample_train_df=subsample_train_df[["path", "participant_id", "sequence_id", "sign"]]
display(subsample_train_df)

# %% [markdown]
# ### EDA

# %% [markdown]
# ##### Data Dictionary (columns and their meanings)
# Path - filepath to the landmark file (parquet)
# participant_id - who the isolated sign event parquet files are for
# sequence_id - one sequence is a single isolated sign that we have to classify (one parquet file for each, 94,477)
# sign - the label for each event

# %%
# Here we're looking at the participant_id column
display(train_df["participant_id"].astype(str).describe().to_frame().T)

print('--------------------------------------------')

participant_count_map = train_df["participant_id"].value_counts().to_dict()
print("1. Number of Unique Participants: ", len(participant_count_map))
print("2. Average Number of Rows Per Participant: ", np.array(list(participant_count_map.values())).mean())
print("3. Standard Deviation in Counts Per Participant: ", np.array(list(participant_count_map.values())).std())
print("4. Minimum Number of Examples For One Participant: ", np.array(list(participant_count_map.values())).min())
print("5. Maximum Number of Examples For One Participant: ", np.array(list(participant_count_map.values())).max())

# set participant_id to be a string
train_df["participant_id"] = train_df["participant_id"].astype(str)
subsample_train_df["participant_id"] = subsample_train_df["participant_id"].astype(str)

# %%
# Here we are looking at the 'sign' column
display(train_df["sign"].describe().to_frame().T)

print('-----------------------------------------------')

sign_count_map = train_df["sign"].value_counts().to_dict()
print("1. Number Of Unique Signs: ", len(sign_count_map))
print("2. Average Number of Rows Per Sign: ", np.array(list(sign_count_map.values())).mean())
print("3. Standard Deviation in Counts Per Sign: ", np.array(list(sign_count_map.values())).std())
print("4. Minimum Number of Examples For One Sign: ", np.array(list(sign_count_map.values())).min())
print("5. Maximum Number of Examples For One Sign: ", np.array(list(sign_count_map.values())).max())

# Looks like the data is pretty balanced
# i.e. one sign is not overly represented way more than the others

# %% [markdown]
# #### Now let's look at what data from the sequence parquet files might be important to include in our model

# %% [markdown]
# For each sequence there is:
# 1. start_frame
# 2. end_frame
# 3. total_frames
# 4. face_count
# 5. pose_count
# 6. left_hand_count
# 7. right_hand_count
# 8. x_min
# 9. x_max
# 10. y_min
# 11. y_max
# 12. z_min
# 13. z_max

# %% [markdown]
# ### Data / Feature Engineering

# %%
def get_seq_meta(row, invert_y=True, do_counts=False):
    """Calculates and adds metadata to the given row of sign language event data.
    
    Args:
        row (pandas.core.series.Series): A row of sign language event data containing columns:
            path: The file path to the Parquet file containing the landmark data for the event.
        invert_y (bool, optional): Whether to invert the y-coordinate of each landmark. Defaults to True.
    
    Returns:
        pandas.core.series.Series: The input row with added metadata columns:
            start_frame: The frame number of the first frame in the event.
            end_frame: The frame number of the last frame in the event.
            total_frames: The number of frames in the event.
            face_count: The number of landmarks in the 'face' type. [optional]
            pose_count: The number of landmarks in the 'pose' type. [optional]
            left_hand_count: The number of landmarks in the 'left_hand' type. [optional]
            right_hand_count: The number of landmarks in the 'right_hand' type. [optional]
            x_min: The minimum x-coordinate value of any landmark in the event.
            x_max: The maximum x-coordinate value of any landmark in the event.
            y_min: The minimum y-coordinate value of any landmark in the event.
            y_max: The maximum y-coordinate value of any landmark in the event.
            z_min: The minimum z-coordinate value of any landmark in the event.
            z_max: The maximum z-coordinate value of any landmark in the event.
    """
    # Extract the sign language event data from the Parquet file at the given path
    df = get_sign_df(row['path'], invert_y=invert_y)
    
    # Count the number of landmarks in each type
    type_counts = df['type'].value_counts(dropna=False).to_dict()
    nan_counts  = df.groupby("type")["x"].apply(lambda x: x.isna().sum())
    
    # Calculate metadata for the event and add it to the input row
    row['start_frame'] = df['frame'].min()
    row['end_frame'] = df['frame'].max()
    row['total_frames'] = df['frame'].nunique()
    
    if do_counts:
        for _type in ["face", "pose", "left_hand", "right_hand"]:
            row[f'{_type}_count'] = type_counts[_type]
            row[f'{_type}_nan_count'] = nan_counts[_type]
        
    for coord in ['x', 'y', 'z']:
        row[f'{coord}_min'] = df[coord].min()
        row[f'{coord}_max'] = df[coord].max()
    
    return row

type_kp_map = dict(face=468, left_hand=21, pose=33, right_hand=21)
col_order = [
    'path', 'participant_id', 'sequence_id', 'sign', 'start_frame', 'end_frame', 'total_frames', 
    'face_nan_count', 'face_nan_pct', 'left_hand_nan_count', 'left_hand_nan_pct', 'pose_nan_count', 'pose_nan_pct',
    'right_hand_nan_count', 'right_hand_nan_pct', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
]

LOAD_EXTENDED = False
if not LOAD_EXTENDED:
    # Will take around 5-10 minutes on subsample and around 50-100 minutes on the full dataset
    subsample_train_df = subsample_train_df.progress_apply(lambda x: get_seq_meta(x, do_counts=True), axis=1)
    for _type, _count in type_kp_map.items():
        subsample_train_df[f"{_type}_appears_pct"] = subsample_train_df[f"{_type}_count"]/(subsample_train_df[f"total_frames"]*_count)
        subsample_train_df[f"{_type}_nan_pct"]     = subsample_train_df[f"{_type}_nan_count"]/(subsample_train_df[f"total_frames"]*_count)
    # Extended save for later...
    subsample_train_df.to_csv("extended_train.csv", index=False)
    display(subsample_train_df)
else:
    del subsample_train_df
    for _type, _count in type_kp_map.items():
            train_df[f"{_type}_appears_pct"] = train_df[f"{_type}_count"]/(train_df[f"total_frames"]*_count)
            train_df[f"{_type}_nan_pct"]     = train_df[f"{_type}_nan_count"]/(train_df[f"total_frames"]*_count)
    train_df = train_df[col_order]
    display(train_df)

# %%
train_df


# %% [markdown]
# ### Create TFRecords

# %%
class CFG:
    data_path = r'C:\repo\math4920\ASLKaggleProject\asl-signs\\'
    sequence_length = 12
    rows_per_frame = 543 


# %%
ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset_with_imputation(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    data.replace(np.nan, 0, inplace=True)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dic = json.load(f)
    return dic


# %%
train = pd.read_csv(r"C:\repo\math4920\ASLKaggleProject\asl-signs\train.csv")
label_index = read_dict(r"C:\repo\math4920\ASLKaggleProject\asl-signs\sign_to_prediction_index_map.json")
index_label = dict([(label_index[key], key) for key in label_index])
print(label_index)
train["label"] = train["sign"].map(lambda sign: label_index[sign])
train.head()


# %%
def create_record(feature, label):
    dic = {}
    dic["feature"] = tf.train.Feature(float_list=tf.train.FloatList(value=feature))
    dic["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    record_bytes = tf.train.Example(features=tf.train.Features(feature=dic)).SerializeToString()
    return record_bytes
    
def decode_function(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,
      # Schema
      {
          "feature": tf.io.FixedLenFeature([CFG.sequence_length * CFG.rows_per_frame * 3], dtype=tf.float32),
          "label": tf.io.FixedLenFeature([], dtype=tf.int64)
      }
  )


# %%
cwd = os.getcwd() + r"\asl-signs\tfrecords\\"

for participant_id in train.participant_id.unique():
    df = train[train.participant_id == participant_id]
    save_path = f"{cwd}{participant_id}.tfrecords"
    print(save_path)
    with tf.io.TFRecordWriter(save_path) as file_writer:
        for i in tqdm(range(len(df))):
            path = f"{CFG.data_path}{df.iloc[i].path}"
            feature = load_relevant_data_subset_with_imputation(path)
            feature = tf.image.resize(tf.constant(feature), (CFG.sequence_length, 543)).numpy().reshape(-1)
            label = int(df.iloc[i].label)
            file_writer.write(create_record(feature, label))

# %% [markdown]
# ### Baseline Attempt

# %%
cwd = os.getcwd()
train_x    = np.load(cwd + "/asl-signs/gislr-feature-data/feature_data.npy").astype(np.float32)
train_y    = np.load(cwd + "/asl-signs/gislr-feature-data/feature_labels.npy").astype(np.uint8)
BATCH_SIZE = 64

N_TOTAL = train_x.shape[0]
VAL_PCT = 0.1
N_VAL   = int(N_TOTAL*VAL_PCT)
N_TRAIN = N_TOTAL-N_VAL

random_idxs = random.sample(range(N_TOTAL), N_TOTAL)
train_idxs, val_idxs = np.array(random_idxs[:N_TRAIN]), np.array(random_idxs[N_TRAIN:])

val_x, val_y = train_x[val_idxs], train_y[val_idxs]
train_x, train_y = train_x[train_idxs], train_y[train_idxs]

# train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
#                           .shuffle(N_TRAIN)\
#                           .batch(BATCH_SIZE, drop_remainder=True)\
#                           .prefetch(tf.data.AUTOTUNE)

# val_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
#                           .shuffle(N_VAL)\
#                           .batch(BATCH_SIZE, drop_remainder=True)\
#                           .prefetch(tf.data.AUTOTUNE)

# train_ds, val_ds

# %%
def fc_block(inputs, output_channels, dropout=0.2):
    x = tf.keras.layers.Dense(output_channels)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    return x

def get_model(n_labels=250, init_fc=512, n_blocks=2, _dropout_1=0.2, _dropout_2=0.6, flat_frame_len=3258):
    _inputs = tf.keras.layers.Input(shape=(flat_frame_len,))
    x = _inputs
    
    # Define layers
    for i in range(n_blocks):
        x = fc_block(
            x, output_channels=init_fc//(2**i), 
            dropout=_dropout_1 if (1+i)!=n_blocks else _dropout_2
        )
    
    # Define output layer
    _outputs = tf.keras.layers.Dense(n_labels, activation="softmax")(x)
    
    # Build the model
    model = tf.keras.models.Model(inputs=_inputs, outputs=_outputs)
    return model

model = get_model()
model.compile(tf.keras.optimizers.Adam(0.000333), "sparse_categorical_crossentropy", metrics="acc")
model.summary()

tf.keras.utils.plot_model(model)

# %%
# # !mkdir models

# %%
cb_list = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.8, verbose=1)
]
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=100, callbacks=cb_list, batch_size=BATCH_SIZE)
model.save("./models/asl_model")

# %%
model.evaluate(val_x, val_y)
for x,y in zip(val_x[:10], val_y[:10]):
    print(f"PRED: {decoder(np.argmax(model.predict(tf.expand_dims(x, axis=0), verbose=0), axis=-1)[0]):<20} – GT: {decoder(y)}")


# %%
class PrepInputs(tf.keras.layers.Layer):
    def __init__(self, face_idx_range=(0, 468), lh_idx_range=(468, 489), 
                 pose_idx_range=(489, 522), rh_idx_range=(522, 543)):
        super(PrepInputs, self).__init__()
        self.idx_ranges = [face_idx_range, lh_idx_range, pose_idx_range, rh_idx_range]
        self.flat_feat_lens = [3*(_range[1]-_range[0]) for _range in self.idx_ranges]
    
    def call(self, x_in):
        
        # Split the single vector into 4
        xs = [x_in[:, _range[0]:_range[1], :] for _range in self.idx_ranges]
        
        # Reshape based on specific number of keypoints
        xs = [tf.reshape(_x, (-1, flat_feat_len)) for _x, flat_feat_len in zip(xs, self.flat_feat_lens)]
        
        # Drop empty rows - Empty rows are present in 
        #   --> pose, lh, rh
        #   --> so we don't have to for face
        xs[1:] = [
            tf.boolean_mask(_x, tf.reduce_all(tf.logical_not(tf.math.is_nan(_x)), axis=1), axis=0)
            for _x in xs[1:]
        ]
        
        # Get means and stds
        x_means = [tf.math.reduce_mean(_x, axis=0) for _x in xs]
        x_stds  = [tf.math.reduce_std(_x,  axis=0) for _x in xs]
        
        x_out = tf.concat([*x_means, *x_stds], axis=0)
        x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
        return tf.expand_dims(x_out, axis=0)
    
PrepInputs()(load_relevant_data_subset(train_df.path[0]))

# %%
print(train_df.path[0])

# %%
import tensorflow as tf

class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – a preprocessing model
        – the ISLR model 
    """

    def __init__(self, islr_model):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = PrepInputs()
        self.islr_model   = islr_model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        outputs = self.islr_model(x)[0, :]

        # Return a dictionary with the output tensor
        return {'outputs': outputs}

tflite_keras_model = TFLiteModel(islr_model=model)
demo_output = tflite_keras_model(load_relevant_data_subset(train_df.path[0]))["outputs"]
decoder(np.argmax(demo_output.numpy(), axis=-1))

# %%
# !pip uninstall tflite_runtime

# %%
# !pip install tflite_runtime

# %%
keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
tflite_model = keras_model_converter.convert()
with open(r'C:\repo\math4920\ASLKaggleProject\kaggle\working\models\model.tflite', 'wb') as f:
    f.write(tflite_model)
# # !zip submission.zip /kaggle/working/models/model.tflite


import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter("/kaggle/working/models/model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
# if REQUIRED_SIGNATURE not in found_signatures:
#     raise KernelEvalException('Required input signature not found.')
prediction_fn = interpreter.get_signature_runner("serving_default")

output = prediction_fn(inputs=load_relevant_data_subset(train_df.path[0]))
sign = np.argmax(output["outputs"])

print("PRED : ", decoder(sign))
print("GT   : ", train_df.sign[0])

# %% [markdown]
# #### Sources:
# https://www.kaggle.com/code/dschettler8845/gislr-learn-eda-baseline
#
# https://www.kaggle.com/code/lonnieqin/islr-create-tfrecord
