import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_transform as tft


def predict(data_path='./data/hidden_test.csv',
            model_path='./cached_model/',
            tfx_root='cached_tfx'):
    
    TRANSFORM_GRAPH_PATH = f'./{tfx_root}/pipelines/transform/Transform/transform_graph/4/'
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        ind = 0
        configs = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)]
        tf.config.experimental.set_virtual_device_configuration(gpus[ind], configs)

    custom_model = tf.saved_model.load(model_path)
    tf_transform_output = tft.TFTransformOutput(TRANSFORM_GRAPH_PATH)
    prerocessing = tf_transform_output.transform_features_layer()
    
    dataframe = pd.read_csv(data_path)
    dataset = tf.data.Dataset.from_tensor_slices(dataframe.applymap(lambda x: [x]).to_dict(orient='list'))
    
    def cast(x):
        for k, v in x.items():
            if v.dtype == tf.int32:
                x[k] = tf.cast(v, tf.int64)
        return x
    
    def split(x):
        xc = x.copy()
        if 'target' in xc:
            xc.pop('target')
        return xc
    
    dataset = dataset.map(cast)
    prepared_dataset = dataset.batch(100).map(prerocessing)
    preds = prepared_dataset.map(split).map(custom_model)

    values = []
    for pred in preds:
        values.append(pred.numpy().flatten() * 100)
    values = np.concatenate(values)
    
    dataframe['pred'] = values
    dataframe.to_csv('./result.csv', index=False)
    
    
    