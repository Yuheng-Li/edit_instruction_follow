import json
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--result_jsonl_file_path", type=str, default='result.jsonl', help='')
parser.add_argument("--thres", type=float, default=0.5, help='')
args = parser.parse_args()


data = []
with open(args.result_jsonl_file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line))

total = len(data)
results_TP = 0
results_TN = 0
results_FP = 0
results_FN = 0


thres = args.thres
for datum in data:

    score = datum['score']
    label = datum['label']

    if   score >= thres and label == "Yes":
        results_TP += 1
    elif score <= thres and label == "No":
        results_TN += 1
    elif score >= thres and label == "No":
        results_FP += 1
    elif score <= thres and label == "Yes":
        results_FN += 1
    else:
        assert False


print("Acc: ", (results_TP+results_TN) / total  * 100  )
print(" ")
print("TOTAL: ", total )
print("TP: ", results_TP   )
print("TN: ", results_TN   )
print("FP: ", results_FP   )
print("FN: ", results_FN   )

print("prevision: ", results_TP / (results_TP+results_FP)   )
print("recall: ",    results_TP / (results_TP+results_FN)   )

