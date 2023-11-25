import os
import sys
import subprocess
from nats_bench import create
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from generate_tflite_models import write_models


def main(batch_size, dataset, loops, inf_temp_file, perf_temp_file):
    api = create('../NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
    models_folder = "models"
    results_folder = "results"
    performance_script = "do_perf.sh"
    idx = 0

    while True:
        more_models = write_models(api, idx, batch_size, dataset, models_folder)

        # Call do_perf.sh shell script
        subprocess.run(["sh", performance_script, models_folder, results_folder, "inf_temp.csv", "perf_temp.csv"])

        # Remove all files in folder except index-history.txt
        for file in os.listdir(models_folder):
            os.remove(os.path.join(models_folder, file))

        if not more_models or (loops is not None and idx >= loops - 1):
            break
        print("inloop")

        idx += 1


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process models and run performance script.")

    # Add arguments
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for processing (default: 50)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use: cifar10, cifar100, ImageNet16-120 (default: 'cifar10')")
    parser.add_argument("--loops", type=int, help="Number of loops to run (default: run until all models are processed)")
    parser.add_argument("--inf_temp_file", type=str, default="inf_temp.csv", help="Inference temp file (default: 'inf_temp.csv')")
    parser.add_argument("--perf_temp_file", type=str, default="perf_temp.csv", help="Performance temp file (default: 'perf_temp.csv')")

    # Parse the arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args.batch_size, args.dataset, args.loops, args.inf_temp_file, args.perf_temp_file)