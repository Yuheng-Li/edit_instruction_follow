import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return precision, recall, accuracy

def plot_precision_recall_curve(data_list, labels, thresholds, output_path):
    plt.figure()

    for data, label in zip(data_list, labels):
        precisions = []
        recalls = []

        for threshold in thresholds:
            precision, recall, _ = calculate_precision_recall(data, threshold)
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


def nice_print_pr(data, label):
    precision, recall, accuracy = calculate_precision_recall(data, 0.5)
    print('- - - - - - - - - - - - - - - ')
    print(label)
    print("Precision:", precision*100)
    print("Recall:", recall*100)
    print("Accuracy:", accuracy*100)
    print(' ')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_jsonl_file_paths", type=str, default='result.jsonl', help='Comma-separated paths to the JSONL result files')
    parser.add_argument("--job_type", type=str, default='pr_curve', help='pr_curve, pr')
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

    if args.job_type == 'pr_curve':
        # In this case, it allow multi result files (i.e., len(data_list)>1 is okay )
        thresholds = np.arange(0, 1.01, 0.01)
        plot_precision_recall_curve(data_list, labels, thresholds, args.output_path)
    
    elif args.job_type == 'pr':
        # In this case, we only support one result file. And we will report category PR if exists 
        assert len(data_list) == 1
        data = data_list[0]
        nice_print_pr(data, 'whole data')

        if 'edit_type' in data[0]:
            # Group data based on 'edit_type'
            grouped_data = {}
            for item in data:
                edit_type = item['edit_type']
                if edit_type not in grouped_data:
                    grouped_data[edit_type] = []
                grouped_data[edit_type].append(item)

            for key in grouped_data:
                temp = grouped_data[key]
                nice_print_pr(temp, str(key)+'  data num:'+str(len(temp)))


