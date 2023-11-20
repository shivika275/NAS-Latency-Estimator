from nats_bench import create
from nats_bench.api_utils import time_string
import numpy as np
from SMR_tflite import get_tflite_model
import os


def write_models(api, idx, batch_size, dataset, folder, log='index-history.txt'):
    '''
    Get tflite models 
    '''
    if(not(os.path.exists(folder))):
        os.makedirs(folder)
    
    total_models = len(api.meta_archs)
    start = idx * batch_size
    end =  np.min((total_models, (idx+1) * batch_size ))
    
    log_string = f"Run log: Dataset - {dataset}, Batch_Number - {str(idx)}, Batch_Size - {str(batch_size)}, Start Index Model - {str(start)}, End Index Model - {str(end)}\n"
    
    with open(os.path.join(folder, log), 'a') as file:
        file.write(log_string)
        
    for i in range(start, end):
        tflite_model = get_tflite_model(api, i, dataset)
        name = dataset+'-'+str(i)+'.tflite'
        model_path = os.path.join(folder, name)
        
        with open(model_path, 'wb') as f:
            f.write(tflite_model)


if __name__ == "__main__":
    folder = '/Users/shivikasingh/Desktop/SMR/Project/NAS-Latency-Estimator/NATS-Bench/tflite_models'
    idx = 7
    dataset = 'cifar100' # Options: cifar10, cifar100, ImageNet16-120
    api = create('NATS-Bench/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
    write_models(api, idx, batch_size=50, dataset=dataset, folder=folder)
    