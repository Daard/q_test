
import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    
    schema = {'big_columns' : [str(x) for x in list(range(6))] + [str(x) for x in list(range(9, 13))], 
              'medium_columns' : [str(x) for x in list(range(13, 53))], 
              'small_columns': ['7'],
              }
    outs = {}
    
    for column_type in ['big_columns', 'medium_columns', 'small_columns']:
        columns = []
        for c in schema[column_type]:
            columns.append(inputs[c])                          
        outs[column_type] = tft.scale_by_min_max(tf.concat(columns, axis=-1),
                                                 output_min=-0.5,
                                                 output_max=0.5,
                                                 elementwise=True)
    
    outs['categorical'] = tf.cast(inputs['8'], tf.int64)
    outs['z_normal'] = tft.scale_to_z_score(10 - inputs['6'])
    outs['target'] = inputs['target']
        
    return outs
