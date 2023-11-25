import re
import sys
import os

def parse_simpleperf_report(report_file):
    data = {}
    with open(report_file, 'r') as file:
        lines = file.readlines()
    
    current_event = None
    for line in lines:
        if 'Event:' in line:
            current_event = line.split()[1]
        elif 'Samples:' in line and current_event:
            data[current_event + '-samples'] = line.split()[1]
        elif 'Event count:' in line and current_event:
            data[current_event + '-count'] = line.split()[2]

    return data

def write_to_csv(model_name, data, csv_file):
    with open(csv_file, 'a') as file:
        line = f"{model_name}"
        for event in ['cpu-cycles', 'cache-references', 'cache-misses', 'instructions']:
            line += f",{data.get(event + '-samples', '')},{data.get(event + '-count', '')}"
        file.write(line + '\n')

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 script.py <simpleperf_report_file> <csv_file>")
        sys.exit(1)

    report_file = sys.argv[1]
    csv_file = sys.argv[2]
    model_name = os.path.basename(report_file).replace('cache_report-', '').split('.')[0]

    data = parse_simpleperf_report(report_file)
    write_to_csv(model_name, data, csv_file)

if __name__ == "__main__":
    main()
