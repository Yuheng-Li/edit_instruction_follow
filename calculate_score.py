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

def plot_precision_recall_curve(data_list, labels, thresholds, output_path):
    plt.figure()

    for data, label in zip(data_list, labels):
        precisions = []
        recalls = []

        for threshold in thresholds:
            precision, recall = calculate_precision_recall(data, threshold)
            precisions.append(precision)
            recalls.append(recall)

        # Sort precision and recall based on recall values
        sorted_indices = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_indices]
        precisions = np.array(precisions)[sorted_indices]

        # Calculate AUC-PR
        auc_pr = np.trapz(precisions, recalls)

        plt.plot(recalls, precisions, marker='.', label=f'{label} (AUC-PR = {auc_pr:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    print(f'Precision-Recall curve saved to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_jsonl_file_paths", type=str, help='Comma-separated paths to the JSONL result files')
    parser.add_argument("--output_path", type=str, default='precision_recall_curve.png', help='Path to save the precision-recall curve image')
    args = parser.parse_args()

    # Read data from the JSONL files
    data_list = []
    labels = []
    file_paths = args.result_jsonl_file_paths.split(',')
    for file_path in file_paths:
        data = []
        with open(file_path.strip(), 'r') as file:
            for line in file:
                data.append(json.loads(line))
        data_list.append(data)
        labels.append(file_path.strip())

    # Define a range of thresholds
    thresholds = np.arange(0, 1.01, 0.01)

    # Plot and save the precision-recall curves
    plot_precision_recall_curve(data_list, labels, thresholds, args.output_path)
