# FSI - Project 2

import json
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean

def get_metrics(json_el, hyperparam, metric, average='macro'):
    metrics_data = {}

    if metric=='accuracy':
        for fold, args in json_el.items():
            for hyperparam_obj in args:
                if hyperparam_obj['args'][hyperparam] in metrics_data:
                    metrics_data[hyperparam_obj['args'][hyperparam]].append(hyperparam_obj['metrics'][metric])
                else:
                    metrics_data[hyperparam_obj['args'][hyperparam]] = [hyperparam_obj['metrics'][metric]]
    else:
        for fold, args in json_el.items():
            for hyperparam_obj in args:
                if hyperparam_obj['args'][hyperparam] in metrics_data:
                    metrics_data[hyperparam_obj['args'][hyperparam]].append(hyperparam_obj['metrics'][metric][average])
                else:
                    metrics_data[hyperparam_obj['args'][hyperparam]] = [hyperparam_obj['metrics'][metric][average]]

    return metrics_data


##  Gets data to create a graph to min sample split hyperparam
with open('../results/json/min_sample_split.json', 'r') as myfile:
    data = myfile.read()
min_sample_split_json = json.loads(data)

accuracy_data = get_metrics(min_sample_split_json, 'min_samples_split', 'accuracy')
f1_data = get_metrics(min_sample_split_json, 'min_samples_split', 'f1_score')
precision_data = get_metrics(min_sample_split_json, 'min_samples_split', 'precision')





# Plot accuracy - data
df = pd.DataFrame({'x': list(accuracy_data.keys()),
                   'accuracy': list(map(mean, accuracy_data.values())),
                   'f1_score': list(map(mean, f1_data.values())),
                   'precision': list(map(mean, precision_data.values()))
                   })
# Plot accuracy - lines
plt.plot('x', 'accuracy', data=df, color='red', label='Accuracy')
plt.plot('x', 'f1_score', data=df, color='green', label='F1 Score')
plt.plot('x', 'precision', data=df, color='blue', label='Precision')

# Plot accuracy - create image
plt.legend()
plt.title('Graph to differents metricts for "Min Samples Split" hyperparam')
plt.xlabel('hyperparam value')
plt.ylabel('metric value')
plt.show()
# plt.savefig('../results/graphs/accuracy_graph')