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


def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
@torch.inference_mode()
def measure(model, img_size, num_repeats=50, num_warmup=10):
    # model.cuda()
    model.eval()

    # backbone = model.backbone
    inputs = torch.randn(4, 3, img_size, img_size)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    latencies = []
    for k in range(num_repeats + num_warmup):
        start = cuda_time()
        model(inputs)
        if k >= num_warmup:
            latencies.append((cuda_time() - start) * 1000)

    #latencies = itertools.chain(dist.allgather(latencies))
    latencies = sorted(latencies)

    drop = int(len(latencies) * 0.25)
    return np.mean(latencies[drop:-drop])

def latency(api_main,i,dataset='cifar10'):
  config = api.get_net_config(i, dataset)
  network = get_cell_based_tiny_net(config)
  info = api.get_more_info(i, dataset)
  cost = api.get_cost_info(i,dataset)
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  network = network.to(device)
  if(dataset == 'ImageNet16-120'):
    img_size = 128
  else:
    img_size = 32
  latency = measure(network,img_size)
  data=[config['name'],config['C'],config['N'],config['arch_str'],info['test-accuracy'],info['test-all-time'],info['test-per-time'],info['train-accuracy'],info['train-all-time'],info['train-per-time'],cost['flops'],cost['latency'], cost['params'],cost['T-ori-test@epoch'], cost['T-ori-test@total'], cost['T-train@epoch'], cost['T-train@total'], latency]
  return data




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

# Convert memory size to GB


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
