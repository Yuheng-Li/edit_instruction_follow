import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product

"requested by cherry. the idea she talked with me afternoon for yufan"

def generate_pairs(data):
    yes_items = [item for item in data if item['label'] == 'Yes']
    no_items = [item for item in data if item['label'] == 'No']    
    pairs = list(product(yes_items, no_items))
    return pairs


def has_both_labels(data):
    labels = set(item['label'] for item in data)
    return 'Yes' in labels and 'No' in labels


def group_by_folder_name(data):
    grouped_data = defaultdict(list)
    for item in data:
        folder_name = item['image'][0].split('/')[0]
        grouped_data[folder_name].append(item)
    return [ grouped_data[k] for k in  grouped_data ] # only return value 


def filter_grouped_data(grouped_data):
    grouped_data_new = []
    for data in grouped_data:
        if len(data) > 1 and has_both_labels(data):
            grouped_data_new.append(data) 
    return grouped_data_new


def calculate_acc(test_data_pairs, thres=0.0):

    corr_count = 0
    for test_data in test_data_pairs:
        assert test_data[0]['label'] == "Yes" and test_data[1]['label'] == "No"
        
        yes_pred = test_data[0]['score']
        no_pred =  test_data[1]['score']

        if yes_pred - no_pred > thres:
            corr_count += 1 
    
    return corr_count / len(test_data_pairs)

def main(args):

    # - - - - - - - - - - - - - Read file - - - - - - - - - - - - - #  
    data = []
    with open(args.file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # group data within the same folder and filter them
    grouped_data = group_by_folder_name(data)
    grouped_data = filter_grouped_data(grouped_data)

    # Generate all possible pairs as data
    test_data_pairs = []
    for data in grouped_data:
        test_data_pairs += generate_pairs(data) 
    

    # cal acc 
    thresholds = np.arange(0, 1.001, 0.001)
    accuracys = []
    for thres in thresholds:
        acc = calculate_acc(test_data_pairs, thres)
        accuracys.append( acc )


    # Plot accuracy vs. threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracys, label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='result.jsonl', help='Comma-separated paths to the JSONL result files')
    parser.add_argument("--output_path", type=str, default='acc_thres_curve.png', help='Path to save the precision-recall curve image')
    args = parser.parse_args()

    main(args)