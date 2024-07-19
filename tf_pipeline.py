import tfx
from tfx.components import CsvExampleGen
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.transform.component import Transform
from tfx.v1.dsl import Pipeline 
from tfx.proto import example_gen_pb2

import os

PIPELINE_NAME = "transform_data"
ROOT = './'

DATA_ROOT = os.path.join(ROOT, 'data/')
TF_DATA_ROOT = os.path.join(DATA_ROOT, 'data_for_tfx/')
TRANSFORM_FILE = os.path.join(ROOT, 'transform.py')
TFX_ROOT = os.path.join(ROOT, 'new_tfx')


def create_pipeline(pipeline_name: str = PIPELINE_NAME,
                    data_root: str = TF_DATA_ROOT,
                    transform_module_file: str = TRANSFORM_FILE,
                    train_pattern: str = 'train_data-*',
                    val_pattern: str = 'val_data-*'):
    """Implements the online news pipeline with TFX.""" 
    # Brings data into the pipeline or otherwise joins/converts training data.
    input_cnf = example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(name='train', 
                                                pattern=train_pattern),
                    example_gen_pb2.Input.Split(name='eval', 
                                                pattern=val_pattern)
                ])
    example_gen = CsvExampleGen(input_base=data_root,
                                   input_config=input_cnf)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=transform_module_file)
  
    components = [
          example_gen,
          statistics_gen,
          schema_gen,
          transform
      ]
    
    pipeline_root = os.path.join(TFX_ROOT, 'pipelines', pipeline_name)
    metadata_path = os.path.join(TFX_ROOT, 'metadata', pipeline_name, 'metadata.db')

    sql_cnf = tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path)
    return Pipeline(
              pipeline_name=pipeline_name,
              pipeline_root=pipeline_root,
              metadata_connection_config=sql_cnf,
              enable_cache=True,
              components=components)