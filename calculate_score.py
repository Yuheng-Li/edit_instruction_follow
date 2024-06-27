import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def calculate_precision_recall(data, threshold):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for datum in data:
        score = datum['score']
        label = datum['label']

        if score >= threshold and label == "Yes":
            TP += 1
        elif score < threshold and label == "No":
            TN += 1
        elif score >= threshold and label == "No":
            FP += 1
        elif score < threshold and label == "Yes":
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall

def plot_precision_recall_curve(data, thresholds, output_path):
    precisions = []
    recalls = []

    for threshold in thresholds:
        precision, recall = calculate_precision_recall(data, threshold)
        precisions.append(precision)
        recalls.append(recall)

    plt.figure()
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_jsonl_file_path", type=str, default='result.jsonl', help='Path to the JSONL result file')
    parser.add_argument("--output_path", type=str, default='precision_recall_curve.png', help='Path to save the precision-recall curve image')
    args = parser.parse_args()

    # Read data from the JSONL file
    data = []
    with open(args.result_jsonl_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Define a range of thresholds
    thresholds = np.arange(0, 1.01, 0.01)

    # Plot and save the precision-recall curve
    plot_precision_recall_curve(data, thresholds, args.output_path)
