import os
from typing import List, Tuple
import re

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_transform as tft
from tfx.v1.orchestration import LocalDagRunner
from tensorflow.keras.callbacks import ModelCheckpoint, \
    TerminateOnNaN, TensorBoard, Callback, ProgbarLogger

from tf_pipeline import create_pipeline
from models import CustomModel

LOGS = 'logs'
CHECKPOINTS = 'checkpoints'


def build_callbacks(name, monitor='loss', mode='min',
                    add_callbacks: List[Callback] = None,
                    run_dir: str = None):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        ind = 0
        configs = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
        tf.config.experimental.set_virtual_device_configuration(gpus[ind], configs)
        
    run_dir = './' if run_dir is None else run_dir
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS)
    logs_dir = os.path.join(run_dir, LOGS)
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)

    full_path = os.path.join(checkpoints_dir, name)

    model_checkpoint = ModelCheckpoint(
        filepath=full_path + '_epoch-{epoch:02d}_' + monitor + '-{' + monitor + ':.5f}.h5',
        verbose=1,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        mode=mode,
        save_freq='epoch')

    terminate_on_nan = TerminateOnNaN()

    tensorboard = TensorBoard(log_dir=os.path.join(logs_dir, name),
                              write_graph=True, histogram_freq=3)
    callbacks = [terminate_on_nan, tensorboard, model_checkpoint]

    if add_callbacks is not None:
        callbacks += add_callbacks

    return callbacks


def transform_data(train_data_path='./data/train.csv'):
    data = pd.read_csv(train_data_path)
    train, val = train_test_split(data, random_state=42, test_size=0.2)
    data_for_tfx = './data_for_tfx'
    if not os.path.isdir(data_for_tfx):
        os.mkdir(data_for_tfx)
    train.to_csv(f'{data_for_tfx}/train.csv', index=False)
    val.to_csv(f'{data_for_tfx}/val.csv', index=False)
    
    pipe = create_pipeline(pipeline_name='transform',
                       data_root=data_for_tfx,
                       train_pattern = 'train*',
                       val_pattern = 'val*',
                       transform_module_file='./transform.py')
    LocalDagRunner().run(pipe)
    

def prepare_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    name = 'transform'
    tr_dir = f'./new_tfx/pipelines/{name}/Transform'
    tr_id = 4

    output_dir = os.path.join(tr_dir, f'transform_graph/{tr_id}/')
    tf_transform_output = tft.TFTransformOutput(output_dir)
    feature_spec = tf_transform_output.transformed_feature_spec()

    train_file_pattern = os.path.join(tr_dir,
                                      f'transformed_examples/{tr_id}/Split-train/transformed_examples-00000-of-00001.gz')
    val_file_pattern = os.path.join(tr_dir,
                                    f'transformed_examples/{tr_id}/Split-eval/transformed_examples-00000-of-00001.gz')

    def read_records(file_pattern, feature_spec, batch=50):
        return tf.data.TFRecordDataset(file_pattern, compression_type='GZIP') \
            .map(lambda x: tf.io.parse_example(x, feature_spec)) \
            .batch(batch) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    def split(x):
        xc = x.copy()
        target = xc.pop('target')
        return xc, target / 100.0

    train_dataset = read_records(train_file_pattern, feature_spec).map(split)
    val_dataset = read_records(val_file_pattern, feature_spec).map(split)
    return train_dataset, val_dataset


def train(train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
    tf.keras.backend.clear_session()

    model_name = 'new_model'
    model = CustomModel()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=4e-3,
        decay_steps=10000,
        decay_rate=0.98)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.MeanAbsolutePercentageError()])
    callbacks = build_callbacks(f'{model_name}',
                                monitor='val_mean_absolute_percentage_error',
                                mode='min')

    initial = 0
    total = 20
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=initial,
        epochs=total,
        verbose=1,
        callbacks=callbacks)

    checkpoints_dir = os.path.join('./', 'checkpoints')
    best_metric = 10000
    best_checkpoint = ''
    for file in os.listdir(checkpoints_dir):
        if file[-2:] == 'h5' and file.startswith(model_name):
            match = re.findall('\d{1,3}\.\d{1,5}\.h5', file)[0]
            metric = float(match[:-3])
            if best_metric > metric:
                best_metric = metric
                best_checkpoint = file
    src = os.path.join(checkpoints_dir, best_checkpoint)
    model.load_weights(src)
    tf.saved_model.save(model, model_name)


    
