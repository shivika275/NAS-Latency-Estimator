from nats_bench import create
from nats_bench.api_utils import time_string
import numpy as np
import torch
import xautodl
from xautodl.models import get_cell_based_tiny_net
from xautodl.utils import count_parameters_in_MB
import platform
import GPUtil
import time
from tqdm import tqdm
import pandas as pd
import os
import multiprocessing
import psutil
import tensorflow as tf
from types import new_class
import torch.onnx
import onnx
from onnx_tf.backend import prepare
from onnx import helper

def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def measure(tflite_model, img_size, num_repeats=50, num_warmup=10):
    
    # inputs = torch.randn(4, 3, img_size, img_size)
    inputs = np.random.rand(4, 3, img_size, img_size).astype(np.float32)
    
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    print(input_details)

    # Set the input tensor to the TFLite interpreter
    interpreter.set_tensor(input_details[0]['index'], inputs)

    latencies = []
    for k in range(num_repeats + num_warmup):
        start = cuda_time()
        interpreter.invoke()
        if k >= num_warmup:
            latencies.append((cuda_time() - start) * 1000)

    #latencies = itertools.chain(dist.allgather(latencies))
    latencies = sorted(latencies)

    drop = int(len(latencies) * 0.25)
    return np.mean(latencies[drop:-drop])

def latency(api,i,dataset='cifar10'):
  config = api.get_net_config(i, dataset)
  network = get_cell_based_tiny_net(config)
  info = api.get_more_info(i, dataset)
  cost = api.get_cost_info(i,dataset)
  
  if(dataset == 'ImageNet16-120'):
    img_size = 128
  else:
    img_size = 32

  # convert torch network to onnx
  onnx_network = convert_to_onnx(network, img_size)
  # convert onnx to tflite
  tf_model = convert_onnx_to_tf_rep(onnx_model=onnx_network)

  latency = measure(tf_model,img_size)
  data=[config['name'],config['C'],config['N'],config['arch_str'],info['test-accuracy'],info['test-all-time'],info['test-per-time'],info['train-accuracy'],info['train-all-time'],info['train-per-time'],cost['flops'],cost['latency'], cost['params'],cost['T-ori-test@epoch'], cost['T-ori-test@total'], cost['T-train@epoch'], cost['T-train@total'], latency]
  print(data)
  return data

def rename_onnx_map(onnx_model):
  # Define a mapping from old names to new names
  name_map = {"input.1": "input_1"}

  # Initialize a list to hold the new inputs
  new_inputs = []

  # Iterate over the inputs and change their names if needed
  for inp in onnx_model.graph.input:
      if inp.name in name_map:
          # Create a new ValueInfoProto with the new name
          new_inp = helper.make_tensor_value_info(name_map[inp.name],
                                                  inp.type.tensor_type.elem_type,
                                                  [dim.dim_value for dim in inp.type.tensor_type.shape.dim])
          new_inputs.append(new_inp)
      else:
          new_inputs.append(inp)

  # Clear the old inputs and add the new ones
  onnx_model.graph.ClearField("input")
  onnx_model.graph.input.extend(new_inputs)

  # Go through all nodes in the model and replace the old input name with the new one
  for node in onnx_model.graph.node:
      for i, input_name in enumerate(node.input):
          if input_name in name_map:
              node.input[i] = name_map[input_name]

  return onnx_model

def convert_to_onnx(torch_network, img_size, model_name="model-test"):

  torch_network.eval()
  dummy_input = torch.randn(4, 3, img_size, img_size)

  onnx_path = model_name+".onnx"
  torch.onnx.export(torch_network, dummy_input, onnx_path)

  onnx_model = onnx.load(onnx_path)

  # to avoid error in conversion from onnx to tf
  onnx_model = rename_onnx_map(onnx_model)
  return onnx_model

def convert_onnx_to_tf_rep(onnx_model, tf_model_path='model-test.pb'):
  tf_rep = prepare(onnx_model)
  tf_rep.export_graph(tf_model_path)
  converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
  tflite_model = converter.convert()
  return tflite_model


api = create('NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
print(len(api.meta_archs))

root_dir = 'output_csv'

if torch.cuda.is_available():
  hw = str(GPUtil.getGPUs()[0].name)
else:
  uname = platform.uname()
  num_cores = multiprocessing.cpu_count()
  memory_info = psutil.virtual_memory()
  total_memory = memory_info.total
  total_memory_gb = total_memory / (1024 ** 3)
  hw = str(uname.machine) +'_'+str(uname.processor) +'_'+str(num_cores) +'_'+str(total_memory_gb)

if(not(os.path.exists(os.path.join(root_dir,hw)))):
  os.makedirs(os.path.join(root_dir,hw))

latency_cifar10 = []
latency_cifar100 = []
latency_imagenet = []

total_models = len(api.meta_archs)
# total_models = 4
for i in tqdm(range(total_models),total=total_models):
  latency_cifar10.append(latency(api,i,'cifar10'))
  latency_cifar100.append(latency(api,i,'cifar100'))
  latency_imagenet.append(latency(api,i,'ImageNet16-120'))


all_cols = ['model', 'C','N','arch_str','test-acc','test-all-time','test-per-time','train-acc','train-all-time','train-per-time','flops','latency-gen','params','T-ori-test@epoch','T-ori-test@total','T-train@epoch','T-train@total','latency']
df_cifar10 = pd.DataFrame(latency_cifar10,  columns=all_cols)
df_cifar100 = pd.DataFrame(latency_cifar100,  columns=all_cols)
df_imagenet = pd.DataFrame(latency_imagenet,  columns=all_cols)

df_cifar10.to_csv(os.path.join(root_dir,hw,'cifar10.csv'))
df_cifar100.to_csv(os.path.join(root_dir,hw,'cifar100.csv'))
df_imagenet.to_csv(os.path.join(root_dir,hw,'imagenet16.csv'))
