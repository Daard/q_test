from typing import Dict, Text
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Embedding, Lambda, Activation, BatchNormalization, Input) 


def batch_block_builder(input_size, block_size):
    
    inputs = Input(shape=(input_size,))
    x1 = Dense(block_size)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    return tf.keras.models.Model(inputs=inputs, outputs=x1) 


class CustomModel(tf.keras.models.Model):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.big_dense = tf.keras.models.Sequential([
            batch_block_builder(10, 20),
            batch_block_builder(20, 20),
            batch_block_builder(20, 5),
            ])
        self.medium_dense = tf.keras.models.Sequential([
            batch_block_builder(40, 80),
            batch_block_builder(80, 80),
            batch_block_builder(80, 20),
            ])
        self.small_dense = tf.keras.models.Sequential([
            batch_block_builder(1, 5),
            batch_block_builder(5, 5),
            batch_block_builder(5, 1),
            ])
        self.z_norm_dense = tf.keras.models.Sequential([
            batch_block_builder(1, 5),
            batch_block_builder(5, 5),
            batch_block_builder(5, 1),
            ])
        self.cat_embedding = tf.keras.models.Sequential([
            Embedding(2, 5, name='cats'),
            Lambda(lambda x: x[:, 0, :]),
            batch_block_builder(5, 5),
            batch_block_builder(5, 5),
            batch_block_builder(5, 1),
        ])
            
        self.pred_submodel = tf.keras.models.Sequential([
            Dense(10, name='reducer'),
            BatchNormalization(),
            Activation('relu'),
            Dense(1, name='preds'),
            ])
    
    def call(self, x: Dict[Text, tf.Tensor], trainig=False) -> tf.Tensor:
           
        features = []
        features.append(self.z_norm_dense(x['z_normal']))
        features.append(self.small_dense(x['small_columns']))
        features.append(self.medium_dense(x['medium_columns']))
        features.append(self.big_dense(tf.cast(x['big_columns'], tf.float32)))
        features.append(self.cat_embedding(x['categorical']))
        
        features = tf.concat(features, axis=-1)
        preds = self.pred_submodel(features)
        
        return preds