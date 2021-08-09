import pandas as pd
from os.path import isfile
from utils import np
import scipy.stats


def count_runtime(dataname, n_classes):
    retrain_networks = np.array([])
    retrain_monitors = np.array([])
    adapt_monitors = np.array([])

    cols = [
        'Total',
        'Until all classes known',
        'Retraining networks',
        'Retraining monitors',
        'Adapting monitors',
        'Idle'
    ]
    for instance in [-1]:
        for run_ind in range(1,6):
            file = f'run_time_{dataname}_box-distance_{n_classes}_{instance}_{run_ind}.csv'
            if isfile(file):
                df = pd.read_csv(file, names=cols, header=None)
                retrain_networks = np.append(retrain_networks, df['Retraining networks'][0])
                retrain_monitors = np.append(retrain_monitors, df['Retraining monitors'][0])
                adapt_monitors = np.append(adapt_monitors, df['Adapting monitors'][0])
    avg_retrain_networks = np.mean(retrain_networks)
    avg_retrain_monitors = np.mean(retrain_monitors)
    avg_adapt_monitors = np.mean(adapt_monitors)
    std_retrain_networks = np.std(retrain_networks)
    std_retrain_monitors = np.std(retrain_monitors)
    std_adapt_monitors = np.std(adapt_monitors)

    ci_l, ci_u = confidence_interval(std_retrain_networks)
    print(
        f'{dataname}: average retraining networks {avg_retrain_networks} with confidence interval ({ci_l}, {ci_u})')
    ci_l, ci_u = confidence_interval(std_retrain_monitors)
    print(
        f'{dataname}: average retraining monitors {avg_retrain_monitors} with confidence interval ({ci_l}, {ci_u})')
    ci_l, ci_u = confidence_interval(std_adapt_monitors)
    print(
        f'{dataname}: average adapting monitors {avg_adapt_monitors} with confidence interval ({ci_l}, {ci_u})')


def confidence_interval(sample_std):
    t_bounds = scipy.stats.t.interval(0.95, df=4)
    ci_l = t_bounds[0] * sample_std / np.sqrt(5)
    ci_u = t_bounds[1] * sample_std / np.sqrt(5)

    return ci_l, ci_u


count_runtime('MNIST', 5)
count_runtime('F_MNIST', 5)
count_runtime('CIFAR', 5)
count_runtime('GTSRB', 22)
count_runtime('EMNIST', 24)
