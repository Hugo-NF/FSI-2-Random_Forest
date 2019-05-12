# FSI - Project 2

import json
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean


# Get metrics(accuracy, f1_score and precision) from json element
def get_metrics(json_el, hyperparam, average='macro'):
    accuracy_data = {}
    f1_score_data = {}
    precision_data = {}

    for fold, args in json_el.items():
        for hyperparam_obj in args:
            if hyperparam_obj['args'][hyperparam] in accuracy_data:
                accuracy_data[hyperparam_obj['args'][hyperparam]].append(hyperparam_obj['metrics']['accuracy'])
                f1_score_data[hyperparam_obj['args'][hyperparam]].append(hyperparam_obj['metrics']['f1_score'][average])
                precision_data[hyperparam_obj['args'][hyperparam]].append(
                    hyperparam_obj['metrics']['precision'][average])
            else:
                accuracy_data[hyperparam_obj['args'][hyperparam]] = [hyperparam_obj['metrics']['accuracy']]
                f1_score_data[hyperparam_obj['args'][hyperparam]] = [hyperparam_obj['metrics']['f1_score'][average]]
                precision_data[hyperparam_obj['args'][hyperparam]] = [hyperparam_obj['metrics']['precision'][average]]

    return list(accuracy_data.keys()), accuracy_data, f1_score_data, precision_data


# Get statistics given a aggregate function, aggregating list of values in which fold following the aggregate function
def get_aggregated_statistics(datas, aggregate=mean):
    ret = {
        'hyperparam_value': datas[0],
        'accuracy': {
            'aggregated_list': list(map(aggregate, datas[1].values()))
        },
        'f1_score': {
            'aggregated_list': list(map(aggregate, datas[2].values()))
        },
        'precision': {
            'aggregated_list': list(map(aggregate, datas[3].values()))
        },
    }

    # max of which metric
    ret['accuracy']['max_value'] = max(ret['accuracy']['aggregated_list'])
    ret['f1_score']['max_value'] = max(ret['f1_score']['aggregated_list'])
    ret['precision']['max_value'] = max(ret['precision']['aggregated_list'])

    # hyperparam_value of which max metric
    ret['accuracy']['max_hyperparam_value'] = ret['hyperparam_value'][
        ret['accuracy']['aggregated_list'].index(ret['accuracy']['max_value'])]
    ret['f1_score']['max_hyperparam_value'] = ret['hyperparam_value'][
        ret['f1_score']['aggregated_list'].index(ret['f1_score']['max_value'])]
    ret['precision']['max_hyperparam_value'] = ret['hyperparam_value'][
        ret['precision']['aggregated_list'].index(ret['precision']['max_value'])]

    return ret


# Plot the graph given a statistics data, and save the graph in image file
def plot_graph(data_statistics, title="", image_name=None, hyperparam_name=""):
    df = pd.DataFrame({'x': data_statistics['hyperparam_value'],
                       'accuracy': data_statistics['accuracy']['aggregated_list'],
                       'f1_score': data_statistics['f1_score']['aggregated_list'],
                       'precision': data_statistics['precision']['aggregated_list']
                       })

    # Plot accuracy - lines
    plt.plot('x', 'accuracy', data=df, color='red', label='Accuracy')
    plt.plot('x', 'f1_score', data=df, color='green', label='F1 Score')
    plt.plot('x', 'precision', data=df, color='blue', label='Precision')

    # Plot accuracy - create image
    plt.legend()
    plt.title(title)
    plt.xlabel('{hyperparam_name} value'.format(hyperparam_name=hyperparam_name))
    plt.ylabel('metric value')

    if image_name:
        plt.savefig('../results/graphs/{name}.png'.format(name=image_name))

    plt.show()


# Open a json file, parse it
# use functions above to generate graph and return the statistics, given a hyperparam
def open_and_plot_graph_for_hyperparam(hyperparam_name, hyperparam_file_name, attribute_name, aggregate=mean):
    with open('../results/json/{file_name}.json'.format(file_name=hyperparam_file_name), 'r') as myfile:
        data = myfile.read()

    json_el = json.loads(data)

    datas = get_metrics(json_el, attribute_name)

    # get statistics of hyperparam
    statistics_data = get_aggregated_statistics(datas, aggregate)

    # Plot hyperparam graph
    plot_graph(statistics_data,
               hyperparam_name=hyperparam_name,
               title='Metrics values to "{hyperparam_name}" hyperparam'.format(hyperparam_name=hyperparam_name),
               image_name=hyperparam_name)

    return statistics_data


# --------------- PLOT AND SAVE STATISTICS OF HYPERPARAMS --------------

#  create a graph to number os estimators hyperparam
n_estimators_statistics = open_and_plot_graph_for_hyperparam(
    hyperparam_name='Number of estimators',
    hyperparam_file_name='n_estimators',
    attribute_name='n_estimators',
    aggregate=min)

#  create a graph to min sample split hyperparam
min_samples_split_statistics = open_and_plot_graph_for_hyperparam(
    hyperparam_name='Min Samples Split',
    hyperparam_file_name='min_samples_split',
    attribute_name='min_samples_split',
    aggregate=min)

#  create a graph to min sample split hyperparam
criterion_statistics = open_and_plot_graph_for_hyperparam(
    hyperparam_name='Criterions',
    hyperparam_file_name='criterion',
    attribute_name='criterion',
    aggregate=min)

#  create a graph to min sample split hyperparam
max_depth_statistics = open_and_plot_graph_for_hyperparam(
    hyperparam_name='Max Depth',
    hyperparam_file_name='max_depth',
    attribute_name='max_depth',
    aggregate=min)

#  create a graph to min samples leaf hyperparam
min_samples_leaf_statistics = open_and_plot_graph_for_hyperparam(
    hyperparam_name='Min Samples Leaf',
    hyperparam_file_name='min_samples_leaf',
    attribute_name='min_samples_leaf',
    aggregate=min)

#  create a graph to number os estimators hyperparam
min_impurity_decrease_statistics = open_and_plot_graph_for_hyperparam(
    hyperparam_name='Min Impurity Decrease',
    hyperparam_file_name='min_impurity_decrease',
    attribute_name='min_impurity_decrease',
    aggregate=min)

# --------------- END PLOT AND SAVE STATISTICS OF HYPERPARAMS --------------

print("Table of bests values")
print('Number of estimators',
      round(n_estimators_statistics['accuracy']['max_value'], 3),
      n_estimators_statistics['accuracy']['max_hyperparam_value'],
      round(n_estimators_statistics['f1_score']['max_value'], 3),
      n_estimators_statistics['f1_score']['max_hyperparam_value'],
      round(n_estimators_statistics['precision']['max_value'], 3),
      n_estimators_statistics['precision']['max_hyperparam_value'],
      sep=','
      )

print('Min Samples Split',
      round(min_samples_split_statistics['accuracy']['max_value'], 3),
      min_samples_split_statistics['accuracy']['max_hyperparam_value'],
      round(min_samples_split_statistics['f1_score']['max_value'], 3),
      min_samples_split_statistics['f1_score']['max_hyperparam_value'],
      round(min_samples_split_statistics['precision']['max_value'], 3),
      min_samples_split_statistics['precision']['max_hyperparam_value'],
      sep=','
      )

print('Criterion',
      round(criterion_statistics['accuracy']['max_value'], 3),
      criterion_statistics['accuracy']['max_hyperparam_value'],
      round(criterion_statistics['f1_score']['max_value'], 3),
      criterion_statistics['f1_score']['max_hyperparam_value'],
      round(criterion_statistics['precision']['max_value'], 3),
      criterion_statistics['precision']['max_hyperparam_value'],
      sep=','
      )

print('Max Depth',
      round(max_depth_statistics['accuracy']['max_value'], 3),
      max_depth_statistics['accuracy']['max_hyperparam_value'],
      round(max_depth_statistics['f1_score']['max_value'], 3),
      max_depth_statistics['f1_score']['max_hyperparam_value'],
      round(max_depth_statistics['precision']['max_value'], 3),
      max_depth_statistics['precision']['max_hyperparam_value'],
      sep=','
      )

print('Min Samples Leaf',
      round(min_samples_leaf_statistics['accuracy']['max_value'], 3),
      min_samples_leaf_statistics['accuracy']['max_hyperparam_value'],
      round(min_samples_leaf_statistics['f1_score']['max_value'], 3),
      min_samples_leaf_statistics['f1_score']['max_hyperparam_value'],
      round(min_samples_leaf_statistics['precision']['max_value'], 3),
      min_samples_leaf_statistics['precision']['max_hyperparam_value'],
      sep=','
      )

print('Min Impurity Decrease',
      round(min_impurity_decrease_statistics['accuracy']['max_value'], 3),
      min_impurity_decrease_statistics['accuracy']['max_hyperparam_value'],
      round(min_impurity_decrease_statistics['f1_score']['max_value'], 3),
      min_impurity_decrease_statistics['f1_score']['max_hyperparam_value'],
      round(min_impurity_decrease_statistics['precision']['max_value'], 3),
      min_impurity_decrease_statistics['precision']['max_hyperparam_value'],
      sep=','
      )
