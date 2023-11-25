from nats_bench import create
from nats_bench.api_utils import time_string
import numpy as np
from SMR_tflite import get_tflite_model
import os
import sys


def write_models(api, idx, batch_size, dataset, folder, log='index-history.txt'):
    '''
    Get tflite models 
    '''
    if(not(os.path.exists(folder))):
        os.makedirs(folder)
    
    more_models = False
    total_models = len(api.meta_archs)
    start = idx * batch_size
    end =  np.min((total_models, (idx+1) * batch_size ))
    if end<total_models:
        more_models = True
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_string = f"Run log: Dataset - {dataset}, Batch_Number - {str(idx)}, Batch_Size - {str(batch_size)}, Start Index Model - {str(start)}, End Index Model - {str(end)}\n"
    
    with open(os.path.join(script_dir, log), 'a') as file:
        file.write(log_string)
        
    for i in range(start, end):
        tflite_model = get_tflite_model(api, i, dataset)
        name = dataset+'-'+str(i)+'.tflite'
        model_path = os.path.join(folder, name)
        
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
    return more_models

def main():
    # Check the number of arguments
    if len(sys.argv) != 5:
        print("Usage: generate_tflite_models.py <folder> <idx> <dataset> <batch_size>")
        sys.exit(1)

    # Assign variables from command-line arguments
    folder = sys.argv[1]
    idx = int(sys.argv[2])
    dataset = sys.argv[3] # Options: cifar10, cifar100, ImageNet16-120
    batch_size = int(sys.argv[4])

    # Create NATS-Bench API
    api = create('NATS-Bench/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

    # Call the write_models function
    write_models(api, idx, batch_size, dataset, folder)

if __name__ == "__main__":
    main()
    