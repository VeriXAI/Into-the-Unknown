import numpy as np

from utils import plot_test_accuracy, plt, color_blind


monitor_types = ['box-distance', 'abstraction', 'alpha', 'random']
labels = ['quantitative', 'abstraction', 'softmax', 'random']
instances = [1]
run_range = range(1, 6)


def plot_all(dataname, n_classes):
    plot_test_accuracy(x_type='monitors', dataname=dataname, instances=instances, run_range=run_range,
                       base_line=False, monitor_types=monitor_types, labels=labels, n_classes=n_classes, save=True)
    plot_test_accuracy(x_type='requests', dataname=dataname, instances=instances, run_range=run_range,
                       base_line=False, monitor_types=monitor_types, labels=labels, n_classes=n_classes, save=True)


# monitor-comparison experiment
plot_all(dataname='MNIST', n_classes=5)
plot_all(dataname='F_MNIST', n_classes=5)
plot_all(dataname='VGG_CIFAR10', n_classes=5)
plot_all(dataname='GTSRB', n_classes=22)
plot_all(dataname='EMNIST', n_classes=24)


# ablation-threshold experiment
monitor_types = ['box-distance']

ax1 = plt.figure(figsize=(8.5 * 0.6, 7.0 * 0.6), frameon=False).add_subplot()
colors = color_blind(3)
plot_test_accuracy(ax=ax1, x_type='monitors', dataname='MNIST', instances=[3], run_range=[1],
                   monitor_types=monitor_types, labels=['static distance threshold'],
                   base_line=False, monitor_markers=['X'], line_styles=['--'], save=False, colors=[colors[1]])
plot_test_accuracy(ax=ax1, x_type='monitors', dataname='MNIST', instances=[2], run_range=[1],
                   monitor_types=monitor_types, labels=['dynamic distance threshold'],
                   base_line=False, monitor_markers=['*'], save=False, colors=[colors[0]])
plot_test_accuracy(ax=ax1, x_type='monitors', dataname='MNIST', instances=[4], run_range=[1],
                   monitor_types=monitor_types, labels=['dynamic dist. threshold & no PCA'],
                   base_line=False, monitor_markers=['P'], line_styles=[':'], save=False, colors=[colors[2]],
                   bbox_to_anchor=(0.1, 0.8))
ax1.legend(fancybox=False, shadow=False, prop={"size": 10}, frameon=True, handlelength=1.5)
_, end = ax1.get_xlim()
plt.xticks(np.arange(0, end, 20000))
plt.savefig(f"MNIST_static_distance_comparison.pdf", dpi=300, bbox_inches='tight', transparent=True)

ax2 = plt.figure(figsize=(8.5 * 0.6, 7.0 * 0.6), frameon=False).add_subplot()
plot_test_accuracy(ax=ax2, x_type='monitors', dataname='MNIST', instances=[5], run_range=[1],
                   monitor_types=monitor_types, labels=['distance threshold=0.7'],
                   base_line=False, monitor_markers=['X'], line_styles=['--'], save=False, colors=[colors[1]])
plot_test_accuracy(ax=ax2, x_type='monitors', dataname='MNIST', instances=[2], run_range=[1],
                   monitor_types=monitor_types, labels=['distance threshold=1.0'],
                   base_line=False, monitor_markers=['*'], save=False, colors=[colors[0]])
plot_test_accuracy(ax=ax2, x_type='monitors', dataname='MNIST', instances=[6], run_range=[1],
                   monitor_types=monitor_types, labels=['distance threshold=1.3'],
                   base_line=False, monitor_markers=['P'], line_styles=[':'], save=False, colors=[colors[2]],
                   bbox_to_anchor=(0.1, 0.8))
ax2.legend(fancybox=False, shadow=False, prop={"size": 10}, frameon=True, handlelength=1.5)
_, end = ax2.get_xlim()
plt.xticks(np.arange(0, end, 20000))
plt.savefig(f"MNIST_precision_distance_comparison.pdf", dpi=300, bbox_inches='tight', transparent=True)
