import re
import csv
import sys

def extract_value(content, pattern):
    """Extracts a value from the content using a regular expression."""
    match = re.search(pattern, content)
    print(pattern, match.group(0), match.group(1))
    return match.group(1) if match else "N/A"

def parse_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    data = {
        'Initialization Time (ms)': extract_value(content, r'Initialized session in ([\d.]+)ms'),
        'First Inference Time (us)': extract_value(content, r'First inference: ([\d.]+)'),
        'Warmup Average Time (us)': extract_value(content, r'Warmup \(avg\): ([\d.]+)'),
        'Inference Average Time (us)': extract_value(content, r'Inference \(avg\): ([\d.]+)'),
        'Memory Footprint Init (MB)': extract_value(content, r'Memory footprint delta.*?init=([\d.]+)'),
        'Memory Footprint Overall (MB)': extract_value(content, r'Memory footprint delta.*?overall=([\d.]+)')
    }
    return data

def write_to_csv(model_name, data, csv_file_path):
    with open(csv_file_path, 'a', newline='') as csv_file:  # 'a' to append data
        writer = csv.writer(csv_file)
        writer.writerow([model_name] + list(data.values()))

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <results_file> <csv_file>")
        sys.exit(1)

    results_file = sys.argv[1]
    csv_file = sys.argv[2]
    model_name = re.sub(r'res-|\.txt', '', results_file.split('/')[-1])

    data = parse_results(results_file)
    write_to_csv(model_name, data, csv_file)

    print(f"Data appended to CSV file: {csv_file}")

if __name__ == "__main__":
    main()
