import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA, FastICA
from os.path import isfile
from utils import DataSpec, color_blind, copy
import scipy.stats as st

from . import *
from utils.Options import *
import scipy.stats


def initialize_subplots(n_subplots, title):
    n_cols = math.ceil(math.sqrt(n_subplots))
    n_rows = math.ceil(n_subplots / n_cols)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    row = 0
    col = -1
    fig.canvas.set_window_title(title)
    fig.suptitle(title)
    return fig, ax, n_cols, row, col


def initialize_single_plot(title):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    return fig, ax


def plot_histograms(monitor_manager, data_train_monitor: DataSpec, layer2all_trained_values):
    if layer2all_trained_values is None:
        return

    labels = data_train_monitor.ground_truths()
    n_plots = 0
    for class_index, box_family in enumerate(monitor_manager.monitors[0].abstraction.abstractions):
        if n_plots >= N_HISTOGRAM_PLOTS_UPPER_BOUND:
            if n_plots == 0:
                print("Skipping histogram plots as requested in the options!".format(N_HISTOGRAM_PLOTS_UPPER_BOUND))
            else:
                print("Skipping the remaining histogram plots!".format(N_HISTOGRAM_PLOTS_UPPER_BOUND))
            break
        boxes = box_family.boxes
        for box in boxes:
            if box.isempty():
                continue
            if n_plots >= N_HISTOGRAM_PLOTS_UPPER_BOUND:
                break
            n_plots += 1
            for layer_index, all_trained_values in layer2all_trained_values.items():
                # plot 1D-histograms of trained values and compare them with the boxes
                fig, ax, n_cols, row, col = initialize_subplots(len(all_trained_values[0]), "Histograms")
                for dim in range(len(all_trained_values[0])):
                    if col < n_cols - 1:
                        col += 1
                    else:
                        row += 1
                        col = 0
                    dimension = []
                    for ind, trained_value in enumerate(all_trained_values):
                        if labels[ind] == class_index:
                            dimension.append(trained_value[dim])
                    ax[row][col].hist(dimension, color='steelblue',
                                      bins=int(np.sqrt(len(all_trained_values[0]))),
                                      edgecolor='black', linewidth=1)
                    ax[row][col].plot([box.low[dim], box.high[dim]], [1, 1], color='red', linewidth=5)
                plt.draw()
                plt.pause(0.0001)


def plot_model_history(history):
    if history is None:
        return

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    fig, ax, _, _, _ = initialize_subplots(2, "History")
    ax = ax[0]
    for l in loss_list:
        ax[0].plot(epochs, history.history[l],
                   label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        ax[0].plot(epochs, history.history[l], 'g',
                   label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Accuracy
    for l in acc_list:
        ax[1].plot(epochs, history.history[l],
                   label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        ax[1].plot(epochs, history.history[l], 'g',
                   label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.draw()
    plt.pause(0.0001)


def plot_images(images, labels, classes, iswarning: bool, monitor_id: int, c_suggested=None):
    if iswarning:
        n = len(images)
        if n > N_PRINT_WARNINGS:
            print("printing only the first {:d} out of {:d} warnings for monitor {:d}".format(
                N_PRINT_WARNINGS, n, monitor_id))
            n = N_PRINT_WARNINGS
        title_list = ["Warnings of monitor {:d}".format(monitor_id)]
        images_list = [images]
        n_list = [n]
    else:
        title_list = ["Novelties detected by monitor {:d}".format(monitor_id),
                      "Novelties not detected by monitor {:d}".format(monitor_id)]
        images_list = [images["detected"], images["undetected"]]
        n_list = []
        for images in images_list:
            n = len(images)
            if n > N_PRINT_NOVELTIES:
                print("printing only the first {:d} out of {:d} novelties for monitor {:d}".format(
                    N_PRINT_NOVELTIES, n, monitor_id))
                n = N_PRINT_NOVELTIES
            n_list.append(n)

    plotted_once = False
    for title, images, n in zip(title_list, images_list, n_list):
        if n == 0:
            continue
        plotted_once = True
        colors = color_blind(max(classes) + 1)  # get_rgb_colors(max(classes) + 1)
        fig, ax, n_cols, row, col = initialize_subplots(n, title)

        for i, image in enumerate(images):
            if iswarning and i >= N_PRINT_WARNINGS:
                break
            if not iswarning and i >= N_PRINT_NOVELTIES:
                break

            if col < n_cols - 1:
                col += 1
            else:
                row += 1
                col = 0
            ax[row][col].axis('off')
            normalized_image = np.clip(image.original_input, 0, 1)
            if len(normalized_image.shape) > 2 and normalized_image.shape[2] == 1:
                normalized_image = normalized_image.reshape((28, 28))
            ax[row][col].imshow(normalized_image)

            # add ground-truth class
            ax[row][col].scatter(-10, -5, color=colors[image.c_ground_truth])
            ax[row][col].annotate(labels[image.c_ground_truth] + " (GT)", (-8, -5))
            # add predicted class
            ax[row][col].scatter(-10, -10, color=colors[image.c_predicted])
            ax[row][col].annotate(labels[image.c_predicted] + " (NN)", (-8, -10))
            if c_suggested is not None:
                # add suggested class
                ax[row][col].scatter(-10, -15, color=colors[c_suggested[i]])
                ax[row][col].annotate(labels[c_suggested[i]] + " (M)", (-8, -15))

    if plotted_once:
        plt.draw()
        plt.pause(0.0001)
        plt.tight_layout()
        plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight', transparent=True)


def plot_monitor_training(monitor, history, iterations, scores, best_scores, fp_list, fn_list, tp_list,
                          class2inertias, score_name, category_title):
    ax = PLOT_MONITOR_TRAINING_AXIS()
    ax.cla()
    for layer in monitor.layers():
        plot_2d_projection(history=history, monitor=monitor, layer=layer, category_title=category_title, ax=ax)

    fig, ax = PLOT_MONITOR_RATES_AXIS()
    fig.canvas.set_window_title("Monitor-training history")

    # plot rates & score
    ax[0].cla()
    ax[0].scatter(iterations, fp_list, marker='^', c="r")
    ax[0].plot(iterations, fp_list, label="false positive rate", c="r", linestyle=":")
    ax[0].scatter(iterations, fn_list, marker='x', c="b")
    ax[0].plot(iterations, fn_list, label="false negative rate", c="b", linestyle="--")
    ax[0].plot(iterations, scores, label=score_name, c="g")
    ax[0].plot(iterations, best_scores, label="best score", c="orange")
    ax[0].set_title('False rates & score of the monitor')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Rates/Score')
    ax[0].legend()

    # adding another y axis does not work correctly
    # ax2 = ax[0].twinx()
    # ax2.cla()
    # ax2.plot(iterations, n_boxes_list, c="y", linestyle="--")
    # ax2.set_ylabel("# boxes", c="y")
    # ax2.legend()

    # plot ROC curve
    ax[1].cla()
    ax[1].scatter(fp_list, tp_list, marker='^', c="r")
    ax[1].plot([0, 1], [0, 1], label="baseline", c="k", linestyle=":")
    ax[1].set_title('ROC curve')
    ax[1].set_xlabel('False positive rate')
    ax[1].set_ylabel('True positive rate')
    ax[1].legend()

    # plot clustering inertias
    ax[2].cla()
    for class_index, inertias in class2inertias.items():
        ax[2].plot(iterations, inertias, label="clustering inertia class {}".format(class_index), linestyle=":")
    ax[2].set_title('Clustering inertias')
    ax[2].set_xlabel('Iteration')
    ax[2].set_ylabel('Inertia')
    ax[2].legend()

    plt.draw()
    plt.pause(0.0001)


def plot_2d_projection(history, monitor, layer, category_title, all_classes, class_label_map: ClassLabelMap, ax=None,
                       dimensions=None, additional_point_lists=None, distance_thresholds=None, distances=None):
    if ax is None:
        ax = plt.figure(figsize=(8.5 * 0.5, 7.0 * 0.5), frameon=False).add_subplot()
    m_id = 0 if monitor is None else monitor.id()
    title = "Projected data & abstractions:\n{}\n(monitor {:d}, layer {:d})".format(category_title, m_id, layer)
    # ax.figure.suptitle(title)
    ax.figure.canvas.set_window_title(title)
    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_color('#344846')
    # ax.spines['bottom'].set_color('#344846')
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    if dimensions is None:
        dimensions = monitor.dimensions(layer)
    x = dimensions[0]
    y = dimensions[1]
    ax.set_xlabel("PC {:d}".format(x + 1), size=12)  # , color='#344846')
    ax.set_ylabel("PC {:d}".format(y + 1), size=12)  # , color='#344846')
    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style('italic')
    ylab.set_style('italic')
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])

    # mapping 'old class -> new class' where new classes are called 0..n-1 for n classes
    label2color_idx = dict()
    known_labels = class_label_map.known_labels()
    j = 0
    for j, label_j in enumerate(known_labels):
        label2color_idx[label_j] = class_label_map.get_class(label_j)
    for label_j in sorted(all_classes):
        if label_j not in known_labels:
            j += 1
            label2color_idx[label_j] = j

    # create mapping 'class -> values'
    label2values = dict()
    for label_j, vj in zip(history.ground_truths, history.layer2values[layer]):
        if label_j in label2values:
            xs, ys = label2values[label_j]
        else:
            xs = []
            ys = []
            label2values[label_j] = (xs, ys)
        xs.append(vj[x])
        ys.append(vj[y])

    n_classes_all = len(all_classes) + (len(additional_point_lists) if additional_point_lists is not None else 0)
    n_classes_known = len(class_label_map)
    colors = color_blind(n_classes_all)  # get_rgb_colors(n_classes_all)
    known_markers = get_markers(n_classes_known)
    marker_size = 10  # 6
    marker_alpha = 0.2  # 0.2

    # plot abstractions
    if monitor is not None:
        for i, ai in enumerate(monitor.abstraction(layer).abstractions()):
            if ai.isempty():
                continue
            color = colors[class_label_map.get_class(i)]
            if distance_thresholds is not None:
                # plot cluster centers and visualize distance thresholds
                ai.plot_distance(dims=dimensions, ax=ax, color=color, threshold=distance_thresholds[i])
            else:
                # plot abstractions
                ai.plot(dims=dimensions, ax=ax, color=color)

    # scatter plot
    novelties = []
    for label_j in sorted(label2values.keys()):
        (xs, ys) = label2values[label_j]
        color = [colors[label2color_idx[label_j]]]
        if label_j in known_labels:
            marker = known_markers[class_label_map.get_class(label_j)]
        else:
            novelties.append((label_j, color, xs, ys))
            continue
        ax.scatter(xs, ys, alpha=marker_alpha, label="class " + str(label_j),
                   edgecolors=color, marker=marker, s=marker_size, facecolors='none')
    # plot novelties last
    for label_j, color, xs, ys in novelties:
        ax.scatter(xs, ys, label="class " + str(label_j) + "*",
                   edgecolors='k', marker=NOVELTY_MARKER, s=marker_size, zorder=3, facecolors='none')

    # plot additional points
    if additional_point_lists is not None:
        for j, additional_points in enumerate(additional_point_lists):
            color = colors[len(all_classes) + j]
            xs = []
            ys = []
            for vj in additional_points:
                xs.append(vj[x])
                ys.append(vj[y])
            if distances:
                d = distances[j]
                label = "rejected d={:.2f}".format(d)
            else:
                label = "rejected"
            ax.scatter(xs, ys, alpha=marker_alpha,
                       label=label, edgecolors=[color],  # c=[color], alpha=1.0
                       # marker=ADDITIONAL_MARKER, s=marker_size, zorder=3)
                       marker='o', s=30, zorder=3, facecolors='none')

    # fit and plot a distribution to the components
    # plot_contour(data_flat=history.layer2values[layer][:, dimensions], clf)

    # plot legend
    # legend = ax.legend(loc='upper center', bbox_to_anchor=(0.25, 1.1), ncol=4, fancybox=True, shadow=True)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=False, shadow=False,
    #                   prop={"size":14}, frameon=False)

    legend = ax.legend(fancybox=True, shadow=False, prop={"size": 10}, frameon=True)
    plt.tight_layout()

    # redefine alpha value in legend
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    #    lh.set_sizes([10])

    #plt.savefig(f"{title}.pdf", dpi=300, bbox_inches='tight', transparent=True)

    plt.draw()
    plt.pause(0.0001)


def plot_zero_point(ax, color, epsilon, epsilon_relative):
    if epsilon > 0 and not epsilon_relative:
        print("Epsilon with zero filtering is ignored in plotting.")
    ax.scatter([0], [0], alpha=1.0, c=[color], marker="$+$")
    plt.draw()
    plt.pause(0.0001)


def plot_interval(ax, p1, p2, color, epsilon, epsilon_relative, is_x_dim):
    if epsilon > 0:
        print("Epsilon with zero filtering is ignored in plotting.")
    if is_x_dim:
        points = [[p1, 0], [p2, 0]]
    else:
        points = [[0, p1], [0, p2]]
    polygon = Polygon(points, closed=True, linewidth=1, edgecolor=color, facecolor="none")
    ax.add_patch(polygon)


def plot_pie_chart_single(ax, tp, tn, fp, fn, n_run):
    # pie chart, where the slices will be ordered and plotted counter-clockwise
    if n_run >= 0:
        sizes = [ratio(tn, n_run), ratio(tp, n_run), ratio(fp, n_run), ratio(fn, n_run)]
    else:
        sizes = [1, 1, 1, 1]

    colors = ["w", "w", "w", "w"]
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=0, colors=colors,
                                      wedgeprops={"edgecolor": [0, .4, .6]},
                                      pctdistance=1.2, labeldistance=1.5)
    plt.setp(autotexts, size=16)
    patterns = [".", "o", "*", "O"]
    for i in range(len(wedges)):
        wedges[i].set_hatch(patterns[i % len(patterns)])
        if i in [2, 3]:
            wedges[i].set_ec([1, 0, 0])

    """nicer labels but they don't work well"
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(sizes[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    """

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    if n_run < 0:
        # plot only the legend
        ax.cla()
        plt.axis('off')
        labels = 'true negatives', 'true positives', 'false positives', 'false negatives'
        ax.legend(wedges, labels, loc="center", handleheight=3)


def _get_binary_pie(t):
    n = t[0] + t[1]
    return [ratio(t[0], n), ratio(t[1], n)]


def plot_novelty_detection(monitors, novelty_wrapper, confidence_thresholds, n_min_acceptance=None, name=None):
    x, xticks = get_xticks_bars(confidence_thresholds)

    if name is None and n_min_acceptance is not None:
        if n_min_acceptance >= 0:
            name = "acceptance {:d}".format(n_min_acceptance)
        else:
            name = "rejection {:d}".format(-n_min_acceptance)
    for monitor in monitors:
        m_id = monitor if isinstance(monitor, int) else monitor.id()
        y = []
        for confidence_threshold in confidence_thresholds:
            novelties = novelty_wrapper.evaluate_detection(m_id, confidence_threshold,
                                                           n_min_acceptance=n_min_acceptance)
            n = len(novelties["detected"])
            d = n + len(novelties["undetected"])
            y.append(ratio(n, d))

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.bar(x, y, color=[0, .4, .6], edgecolor='white', width=0.5)
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Novelties detected [%]")
        ax.set_ylim([0, 100])
        ax.xaxis.set_ticks(xticks)
        if name is None:
            final_name = "{:d}".format(m_id)
        else:
            final_name = name
        title = "Novelty detection (monitor {})".format(final_name)
        fig.suptitle(title)
        ax.figure.canvas.set_window_title(title)

    plt.draw()
    plt.pause(0.0001)


def plot_novelty_detection_given_all_lists(core_statistics_list_of_lists: list, n_ticks, name=""):
    n_monitors = len(core_statistics_list_of_lists)
    n_bars = len(core_statistics_list_of_lists[0])
    for core_statistics_list in core_statistics_list_of_lists:
        assert len(core_statistics_list) == n_bars, "Incompatible list lengths found!"
    # x = [i for i in range(2, len(core_statistics_list) + 2)]
    # xticks = x
    x, xticks = get_xticks_bars([i for i in range(2, len(core_statistics_list_of_lists[0]) + 2)], n=n_ticks,
                                to_float=False)
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 1.0 / float(n_monitors + 1)
    for b in range(n_bars):
        for i, core_statistics_list in enumerate(core_statistics_list_of_lists):
            cs = core_statistics_list[b]
            d = cs.novelties_detected + cs.novelties_undetected
            nd = ratio(cs.novelties_detected, d)
            nu = ratio(cs.novelties_undetected, d)
            x_adapted = x[b] + i * width
            ax.bar(x_adapted, nd, color=[0, .4, .6], edgecolor='white', width=width)
            sums = nd
            ax.bar(x_adapted, nu, bottom=sums, color=[1, 0.6, 0.2], edgecolor='white', width=width)
    ax.set_ylim([0, 100])
    ax.xaxis.set_ticks(xticks)
    title = "Novelty detection {}".format(name)
    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)

    plt.draw()
    plt.pause(0.0001)


# 'monitors' can either be a list of numbers or a list of Monitor objects
def plot_false_decisions(monitors, history: History, confidence_thresholds, n_min_acceptance=None, name=None,
                         title=None):
    d = len(history.ground_truths)
    x, xticks = get_xticks_bars(confidence_thresholds)

    if name is None and n_min_acceptance is not None:
        if n_min_acceptance >= 0:
            name = "acceptance {:d}".format(n_min_acceptance)
        else:
            name = "rejection {:d}".format(-n_min_acceptance)
    for monitor in monitors:
        m_id = monitor if isinstance(monitor, int) else monitor.id()
        if name is None:
            final_name = "{:d}".format(m_id)
        else:
            final_name = name
        y_fn = []
        y_fp = []
        y_tp = []
        y_tn = []
        for confidence_threshold in confidence_thresholds:
            history.update_statistics(m_id, confidence_threshold=confidence_threshold,
                                      n_min_acceptance=n_min_acceptance)
            fn = history.false_negatives()
            fp = history.false_positives()
            tp = history.true_positives()
            tn = history.true_negatives()
            y_fn.append(ratio(fn, d))
            y_fp.append(ratio(fp, d))
            y_tp.append(ratio(tp, d))
            y_tn.append(ratio(tn, d))

        _plot_false_decisions_helper(x, xticks, y_fn, y_fp, y_tp, final_name, title=title)

    plt.draw()
    plt.pause(0.0001)


def _plot_false_decisions_helper(x, xticks, y_fn, y_fp, y_tp, name, name2="", title=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 0.5
    _plot_bars_helper(ax, width, x, y_fn, y_fp, y_tp)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("True positives [blue] / false negatives [orange] / false positives [red]")
    ax.set_ylim([0, 100])
    ax.xaxis.set_ticks(xticks)
    if title is None:
        title = "Decision performance (monitor {}) {}".format(name, name2)
    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)


def _plot_bars_helper(ax, width, x, y_fn, y_fp, y_tp):
    ax.bar(x, y_tp, color=COLOR_blue, edgecolor="white", width=width)
    sums = y_tp
    ax.bar(x, y_fn, bottom=sums, color=COLOR_yellow, edgecolor="white", hatch="x", width=width)
    sums = [_x + _y for _x, _y in zip(sums, y_fn)]
    ax.bar(x, y_fp, bottom=sums, color=COLOR_red, edgecolor='white', hatch=".", width=width)
    # sums = [_x + _y for _x, _y in zip(sums, y_tp)]
    # ax.bar(x, y_tn, bottom=sums, color=[0, 0.9, 0.1], edgecolor='white', width=width)


def plot_false_decisions_given_list(core_statistics_list: list, n_ticks, name="", name2=""):
    d = core_statistics_list[0].get_n()
    # x = [i for i in range(2, len(core_statistics_list) + 2)]
    # xticks = x
    x, xticks = get_xticks_bars([i for i in range(2, len(core_statistics_list) + 2)], n=n_ticks, to_float=False)
    y_fn = []
    y_fp = []
    y_tp = []
    y_tn = []
    for cs in core_statistics_list:
        y_fn.append(ratio(cs.fn, d))
        y_fp.append(ratio(cs.fp, d))
        y_tp.append(ratio(cs.tp, d))
        y_tn.append(ratio(cs.tn, d))

    _plot_false_decisions_helper(x, xticks, y_fn, y_fp, y_tp, name=name, name2=name2)

    plt.draw()
    plt.pause(0.0001)


def plot_false_decisions_given_all_lists(core_statistics_list_of_lists: list, n_ticks, n_bars=None, name=""):
    n_monitors = len(core_statistics_list_of_lists)
    n_bars_reference = len(core_statistics_list_of_lists[0])
    if n_bars is None:
        n_bars = n_bars_reference

    for core_statistics_list in core_statistics_list_of_lists:
        assert len(core_statistics_list) == n_bars_reference, "Incompatible list lengths found!"
    # x = [i for i in range(2, len(core_statistics_list) + 2)]
    # xticks = x
    x, xticks = get_xticks_bars([i for i in range(2, n_bars + 2)], n=n_ticks, to_float=False)
    fig = plt.figure()
    ax = fig.add_subplot()
    width = 1.0 / float(n_monitors + 1)
    for b in range(n_bars):
        for i, core_statistics_list in enumerate(core_statistics_list_of_lists):
            cs = core_statistics_list[b]
            d = cs.get_n()
            y_fn = ratio(cs.fn, d)
            y_fp = ratio(cs.fp, d)
            y_tp = ratio(cs.tp, d)
            # y_tn = ratio(cs.tn, d)
            x_adapted = x[b] + i * width
            _plot_bars_helper(ax, width, x_adapted, [y_fn], [y_fp], [y_tp])
    ax.set_ylim([0, 100])
    ax.xaxis.set_ticks(xticks)
    title = "Decision performance {}".format(name)
    fig.suptitle(title)
    ax.figure.canvas.set_window_title(title)

    plt.draw()
    plt.pause(0.0001)


def plot_false_decisions_legend():
    fig = plt.figure()
    ax = fig.add_subplot()
    title = "Legend"
    ax.figure.suptitle(title)
    ax.figure.canvas.set_window_title(title)
    labels = 'false positives', 'false negatives', 'true positives'
    width = 0.5

    res1 = ax.bar([1], [1], color=COLOR_red, edgecolor='white', hatch=".", width=width)
    res2 = ax.bar([1], [2], bottom=[1], color=COLOR_yellow, edgecolor="white", hatch="x", width=width)
    res3 = ax.bar([1], [3], bottom=[2], color=COLOR_blue, edgecolor="white", width=width)
    ax.cla()
    plt.axis('off')
    ax.legend((res1[0], res2[0], res3[0]), labels, loc="center", handleheight=3)


def get_xticks_bars(confidence_thresholds, n=10, to_float=True):
    # x ticks
    step = int(len(confidence_thresholds) / n)
    x = []
    for confidence_threshold in confidence_thresholds:
        xi = float_printer(confidence_threshold) if to_float else confidence_threshold
        x.append(xi)
    xticks = [x[i * step] for i in range(n)]
    return x, xticks


def plot_decisions_of_two_approaches(monitor1, history1: History, confidence_threshold1: float,
                                     monitor2, history2: History, confidence_threshold2: float,
                                     class_label_map, all_labels):
    # collect data
    m_id1 = monitor1 if isinstance(monitor1, int) else monitor1.id()
    m_id2 = monitor2 if isinstance(monitor2, int) else monitor2.id()
    ground_truths = history1.ground_truths
    predictions = history1.predictions
    results1 = history1.monitor2results[m_id1]
    results2 = history2.monitor2results[m_id2]
    class2category2numbers = dict()
    category2correctness2numbers = {"a1 a2": {True: 0, False: 0}, "a1 r2": {True: 0, False: 0},
                                    "r1 a2": {True: 0, False: 0}, "r1 r2": {True: 0, False: 0}}
    known_category2correctness2numbers = {"a1 a2": {True: 0, False: 0}, "a1 r2": {True: 0, False: 0},
                                          "r1 a2": {True: 0, False: 0}, "r1 r2": {True: 0, False: 0}}
    novel_category2numbers = {"a1 a2": 0, "a1 r2": 0, "r1 a2": 0, "r1 r2": 0}
    n_classes = len(all_labels)
    for class_id in all_labels:
        class2category2numbers[class_id] = {"a1 a2": 0, "a1 r2": 0, "r1 a2": 0, "r1 r2": 0}
    for gt, pd, r1, r2 in zip(ground_truths, predictions, results1, results2) \
            :  # type: int, int, MonitorResult, MonitorResult
        if r1.accepts(confidence_threshold1):
            if r2.accepts(confidence_threshold2):
                category = "a1 a2"
            else:
                category = "a1 r2"
        else:
            if r2.accepts(confidence_threshold2):
                category = "r1 a2"
            else:
                category = "r1 r2"
        class2category2numbers[gt][category] += 1
        category2correctness2numbers[category][gt == pd] += 1
        if gt in class_label_map:
            known_category2correctness2numbers[category][gt == pd] += 1
        else:
            novel_category2numbers[category] += 1

    # create figures
    fig, ax, n_cols, row, col = initialize_subplots(4, "Comparison with confidences {:f} and {:f}".format(
        confidence_threshold1, confidence_threshold2))
    x = ["a1 a2", "a1 r2", "r1 a2", "r1 r2"]

    # plot comparison by classes
    sums = [0 for _ in range(4)]
    colors = get_rgb_colors(n_classes)
    for i, class_id in enumerate(all_labels):
        current = [class2category2numbers[class_id][x[i]] for i in range(4)]
        is_known_class = class_id in class_label_map
        label = "class {:d} ({})".format(class_id, "o" if is_known_class else "n")
        ax[0][0].bar(x, current, bottom=sums, color=colors[i], edgecolor='white', width=0.5, label=label)
        for j in range(4):
            sums[j] += current[j]
    ax[0][0].set_xlabel("By classes")
    ax[0][0].legend()

    # plot comparison by correct/incorrect
    current = [category2correctness2numbers[x[i]][True] for i in range(4)]
    ax[0][1].bar(x, current, color="b", edgecolor='white', width=0.5, label="correct")
    current2 = [category2correctness2numbers[x[i]][False] for i in range(4)]
    ax[0][1].bar(x, current2, bottom=current, color="r", edgecolor='white', width=0.5, label="incorrect")
    ax[0][1].set_xlabel("All classes")
    ax[0][1].legend()

    # plot comparison by correct/incorrect for known classes
    current = [known_category2correctness2numbers[x[i]][True] for i in range(4)]
    ax[1][0].bar(x, current, color="b", edgecolor='white', width=0.5, label="correct")
    current2 = [known_category2correctness2numbers[x[i]][False] for i in range(4)]
    ax[1][0].bar(x, current2, bottom=current, color="r", edgecolor='white', width=0.5, label="incorrect")
    ax[1][0].set_xlabel("Known classes")
    ax[1][0].legend()

    # plot comparison by correct/incorrect for novel classes
    current = [novel_category2numbers[x[i]] for i in range(4)]
    ax[1][1].bar(x, current, color="r", edgecolor='white', width=0.5)
    ax[1][1].set_xlabel("Novel classes")

    plt.draw()
    plt.pause(0.0001)


def plot_2_principal_components(input_data, ax=None, known_classes=None,
                                dimensions=None):
    if ax is None:
        ax = plt.figure().add_subplot()
    title = "Two principal components"
    ax.figure.suptitle(title)
    ax.figure.canvas.set_window_title(title)
    if dimensions is None:
        dimensions = [0, 1]
    x = dimensions[0]
    y = dimensions[1]
    ax.set_xlabel("principal component{:d}".format(x), size=16)
    ax.set_ylabel("principal component{:d}".format(y), size=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    # take input images with their labels
    images = input_data.inputs() / 255.0
    labels = input_data.categoricals()
    # flatten the data into one vector of pixels
    images_flat = images.reshape(-1, np.prod(images.shape[1:]))
    # choose the variance to capture
    pca = PCA(0.99)
    # fit the PCA instance on the images
    pca.fit(images_flat)
    # find PCs that explain 0.9 variance
    print("Number of principal components:", pca.n_components_)
    # put each vector into its pixel column
    feat_cols = ['pixel' + str(i) for i in range(images_flat.shape[1])]
    df_data = pd.DataFrame(images_flat, columns=feat_cols)
    # take first two components that explain the most
    pca_data = PCA(n_components=2)
    # assign true labels to the first two principal components
    principal_components_data = pca_data.fit_transform(df_data.iloc[:, :-1])
    principal_data_df = pd.DataFrame(data=principal_components_data
                                     , columns=['principal component 1', 'principal component 2'])
    # plot the first two principal components
    principal_data_df['class'] = np.where(labels == 1)[1]
    sns.scatterplot(x='principal component 1', y='principal component 2'
                    , hue='class', data=principal_data_df, alpha=0.3)

    # ax.legend()
    plt.draw()
    plt.pause(0.0001)

    # return transformed data based on the pca.fit
    img_pca = pca.transform(images_flat)
    img_projected = pca.inverse_transform(img_pca)
    return img_projected


def plot_contour(data_flat, clf, layers, dimensions):
    # taken from https://github.com/scikit-learn/scikit-learn/blob/master/examples/mixture/plot_gmm_pdf.py
    # display predicted scores by the model as a contour plot
    for layer in layers:
        data = data_flat[layer]
        model = clf[layer]
        x = np.linspace(min(data[:, dimensions[0]]), max(data[:, dimensions[0]]))
        y = np.linspace(min(data[:, dimensions[1]]), max(data[:, dimensions[1]]))
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = - model.score_samples(data)  # XX
        Z = Z.reshape(X.shape)
        CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                         levels=np.logspace(0, 3, 10))
        CB = plt.colorbar(CS, shrink=0.8, extend='both')


def plot_test_accuracy(x_type='inputs', dataname='MNIST', monitor_types=['box-distance'], n_classes=5, run_range=[1],
                       instances=[1], ax=None, colors=None, labels=None, monitor_markers=None, line_styles=None,
                       base_line=False, save=False, bbox_to_anchor=None):
    if ax is None:
        if x_type == 'classes':
            ax = plt.figure(figsize=(8.5 * 0.5, 7.0 * 0.5), frameon=False).add_subplot()  # (8.5, 7.0)
        else:
            ax = plt.figure(figsize=(8.5 * 0.6, 7.0 * 0.6), frameon=False).add_subplot()  # (8.5, 7.0)
    if colors is None:
        colors = color_blind(len(monitor_types))
    if labels is None:
        labels = monitor_types
    marker_size = 8
    # confidence interval coefficient depending on the degrees of freedom
    # 95% confidence interval:
    # t-distribution two-tail for (1-0.95)=0.05
    # https://www.statisticshowto.com/tables/t-distribution-table/#two
    # t_alpha = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 24: 2.064}
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['left'].set_color('#4C516D')
    #ax.spines['bottom'].set_color('#4C516D')
    #ax.xaxis.label.set_color('#4C516D')
    #ax.yaxis.label.set_color('#4C516D')
    #ax.tick_params(colors='#4C516D')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.tick_params(direction='in')
    if x_type in ['inputs', 'requests', 'monitors', 'acc_monitors']:
        ax.set_xlabel("Processed inputs", size=12, fontname="sans-serif")
        if not monitor_markers:
            monitor_markers = get_markers(len(monitor_types))
    if x_type in ['classes']:
        ax.set_xlabel("Number of known classes", size=12, fontname="sans-serif")
        if not monitor_markers:
            monitor_markers = get_markers(len(monitor_types))
    if x_type in ['requests']:
        ax.set_ylabel("Authority request rate", size=12, fontname="sans-serif")
    elif x_type in ['monitors']:
        ax.set_ylabel("Monitor precision", size=12, fontname="sans-serif")
    elif x_type in ['acc_monitors']:
        ax.set_ylabel("Monitor F1 score", size=12, fontname="sans-serif")
    else:
        ax.set_ylabel("Model Test Accuracy", size=12, fontname="sans-serif")
    cols = ['No monitoring',
            'No monitoring on all classes',
            'With active monitor',
            'With active monitor on all classes',
            'Offline model', 'Monitor accuracy',
            'fn', 'fp', 'tn', 'tp',
            'Number of known classes',
            'Authority requests',
            'Processed inputs']
    cols_monitor = ['NN',
                    'NNall',
                    'fn', 'fp', 'tn', 'tp',
                    'Monitor precision',
                    'Number of known classes',
                    'Authority requests',
                    'Processed inputs']

    hatches = ["x", "o", "/", "*"]
    for ind in range(len(monitor_types)):
        data = {}
        learned_classes = []
        for instance in instances:
            for run_ind in run_range:
                file = 'testset_classifier_accuracy_{}_{}_{}_{}_{}.csv'.format(dataname, monitor_types[ind], n_classes,
                                                                               instance, run_ind)
                file_monitor = 'precision_with_retraining_{}_{}_{}_{}_{}.csv'.format(dataname, monitor_types[ind],
                                                                                     n_classes,
                                                                                     instance, run_ind)
                if isfile(file) and isfile(file_monitor):
                    df = pd.read_csv(file, names=cols, header=None)
                    df_monitor = pd.read_csv(file_monitor, names=cols_monitor, header=None)
                    max_classes = max(df['Number of known classes'].values)
                    learned_classes.append(max_classes)

                    if x_type == 'classes':
                        xs = df['Number of known classes'].values
                        ys = df['With active monitor'].values
                    # if x_type == 'monitors':
                    #    xs = df['Number of known classes'].values
                    #    ys = df['Monitor'].values
                    if x_type == 'inputs':
                        xs = df['Processed inputs'].values
                        ys = df['With active monitor on all classes'].values
                    if x_type == 'requests':
                        xs = df['Processed inputs'].values
                        ys = df['Authority requests'].values / df['Processed inputs'].values
                        '''
                        if dataname in ['MNIST', 'F_MNIST']:
                            xs = df['Processed inputs'].values[df['Authority requests'].values <= 3000]
                            ys = df['Authority requests'].values[df['Authority requests'].values <= 3000] / \
                                 df['Processed inputs'].values[df['Authority requests'].values <= 3000]
                        if dataname in ['CIFAR']:
                            xs = df['Processed inputs'].values[df['Authority requests'].values <= 4999]
                            ys = df['Authority requests'].values[df['Authority requests'].values <= 4999] / \
                                 df['Processed inputs'].values[df['Authority requests'].values <= 4999]
                        if dataname in ['GTSRB']:
                            xs = df['Processed inputs'].values[df['Authority requests'].values <= 3920]
                            ys = df['Authority requests'].values[df['Authority requests'].values <= 3920] / \
                                 df['Processed inputs'].values[df['Authority requests'].values <= 3920]
                        if dataname in ['EMNIST']:
                            xs = df['Processed inputs'].values[df['Authority requests'].values <= 11279]
                            ys = df['Authority requests'].values[df['Authority requests'].values <= 11279] / \
                                 df['Processed inputs'].values[df['Authority requests'].values <= 11279]
                        '''
                    if x_type == 'monitors':
                        xs = df_monitor['Processed inputs'].values
                        ys = df_monitor['Monitor precision'].values
                    if x_type == 'acc_monitors':
                        xs = df['Processed inputs'].values
                        # f1-score: tp/(tp+1/2(fp+fn))
                        ys = df['tp'].values/(df['tp'].values+1/2*(df['fp'].values+df['fn'].values))#ys = df['Monitor accuracy'].values

                    for x, y in zip(xs, ys):
                        if not data:
                            data[x] = [y]
                        else:
                            if x in data.keys():
                                data[x].append(y)
                            else:
                                data[x] = [y]

        if data:
            # compute mean and confidence interval
            if x_type in ['classes', 'monitors']:
                x = sorted([dx for dx in data])
            else:
                x = sorted([dx for dx in data if len(data[dx]) == len(run_range) * len(instances)])
            sample_mean = [np.mean(data[dx]) for dx in x]
            if len(run_range) >= 5 and x_type in ['classes', 'requests', 'inputs', 'monitors', 'acc_monitors']:
                degrees_of_freedom = len(run_range) * len(instances) - 1
                # 95% confidence interval
                # t-distribution two-tail for (1-0.95)=0.05
                # https://www.statisticshowto.com/tables/t-distribution-table/#two
                # alpha = t_alpha[degrees_of_freedom]
                sample_std = [np.std(data[dx], ddof=1) for dx in x]
                t_bounds = scipy.stats.t.interval(0.95, degrees_of_freedom)
                # ci = [alpha * s / np.sqrt(len(run_range) * len(instances)) for s in sample_std]
                ci_l = [t_bounds[0] * s / np.sqrt(len(run_range) * len(instances)) for s in sample_std]
                ci_u = [t_bounds[1] * s / np.sqrt(len(run_range) * len(instances)) for s in sample_std]
                lb = [s + c for s, c in zip(sample_mean, ci_l)]
                ub = [min(s + c, 1.0) for s, c in zip(sample_mean, ci_u)]
                print(
                    f'{monitor_types[ind]} for {dataname}: maximum average accuracy for total {max(learned_classes)} classes learned is {sample_mean[-1]} with confidence interval ({lb[-1] - sample_mean[-1]}, {ub[-1] - sample_mean[-1]})')

            # plot average
            if line_styles == None:
                if monitor_types[ind] != 'box-distance':
                    linestyle = '--'
                else:
                    linestyle = '-'
            else:
                linestyle = line_styles[ind]
            if x_type in ['classes']:
                retrain = x
            else:
                retrain = df['With active monitor on all classes'].values
            ax.plot(x, sample_mean,
                    linestyle=linestyle,
                    label=labels[ind],
                    markeredgewidth=1.5,
                    c=colors[ind], marker=monitor_markers[ind],
                    linewidth=1.5, markersize=marker_size, markerfacecolor=(0, 0, 0, 0),
                    markeredgecolor=colors[ind],
                    markevery=[t for t in range(1, len(retrain) - 1)
                               if retrain[t - 1] != retrain[t]])  # markevery)
            if len(run_range) >= 5 and x_type in ['classes', 'requests', 'inputs', 'monitors', 'acc_monitors']:
                # plot confidence bands
                ax.fill_between(x, lb, ub, color=colors[ind], alpha=.1)
                # ax.errorbar(np.array(x), np.array(sample_mean), yerr=np.array(sample_std))
            if x_type in ['inputs', 'requests', 'monitors', 'acc_monitors']:
                start, end = ax.get_xlim()
                if dataname == 'EMNIST':
                    plt.xticks(np.arange(0, end, 20000))
                else:
                    plt.xticks(np.arange(0, end, 10000))
    if base_line:
        offline = {}
        static = {}
        data = {}
        for instance in instances:
            for run_ind in run_range:
                file = 'testset_classifier_accuracy_{}_{}_{}_{}_{}.csv'.format(dataname, 'box-distance', n_classes,
                                                                               instance, run_ind)
                if isfile(file):
                    if x_type in ['classes']:
                        static_y = df['No monitoring'].values
                    else:
                        static_y = df['No monitoring on all classes'].values
                    offline_y = df['Offline model'].values
                    for x, y, yoff, yst in zip(xs, ys, offline_y, static_y):
                        if not data and not offline and not static:
                            data[x] = [y]
                            offline[x] = [yoff]
                            static[x] = [yst]
                        else:
                            if x in data.keys() and x in offline.keys() and x in static.keys():
                                data[x].append(y)
                                offline[x].append(yoff)
                                static[x].append(yst)
                            else:
                                data[x] = [y]
                                offline[x] = [yoff]
                                static[x] = [yst]
        if x_type in ['classes']:
            x = sorted([dx for dx in data])
        else:
            x = sorted([dx for dx in data if len(data[dx]) == len(run_range) * len(instances)])
        y = [np.mean(offline[dx]) for dx in x]
        ax.plot(x, y, c='#4C516D', linestyle=':',
                label='static full', linewidth=1.5)
        y = [np.mean(static[dx]) for dx in x]
        ax.plot(x, y, c='#4C516D', linestyle='-.',
                markersize=marker_size,
                label='static half', linewidth=1.5)
    if bbox_to_anchor is None:
        bbox_to_anchor = (0.65, 0.07)
        if x_type == 'classes':
            bbox_to_anchor = (0.01, 0.01)
        elif x_type == 'requests':
            if dataname in ['F_MNIST']:
                bbox_to_anchor = (0.6, 0.3)
            elif dataname in ['VGG_CIFAR10']:
                bbox_to_anchor = (0.6, 0.4)
            elif dataname in ['EMNIST']:
                bbox_to_anchor = (0.6, 0.65)
            else:
                bbox_to_anchor = (0.6, 0.6)
        elif x_type == 'monitors':
            if dataname in ['MNIST']:
                bbox_to_anchor = (0.6, 0.1)
            elif dataname in ['F_MNIST']:
                bbox_to_anchor = (0.6, 0.7)
            elif dataname in ['GTSRB']:
                bbox_to_anchor = (0.6, 0.3)
            elif dataname in ['EMNIST']:
                bbox_to_anchor = (0.6, 0.01)
        elif x_type == 'inputs':
            if dataname in ['MNIST']:
                bbox_to_anchor = (0.6, 0.4)
            elif dataname in ['F_MNIST']:
                bbox_to_anchor = (0.45, 0.1)
            elif dataname in ['VGG_CIFAR10']:
                bbox_to_anchor = (0.6, 0.6)
            elif dataname in ['GTSRB']:
                bbox_to_anchor = (0.2, 0.01)

    ax.legend(fancybox=False, shadow=False, prop={"size": 10}, frameon=True,
              loc='lower left', bbox_to_anchor=bbox_to_anchor, handlelength=1.5)
    #for text in ax.legend().get_texts():
    #    text.set_color("#4C516D")
    plt.tight_layout()
    if save:
        plt.savefig(f"{dataname}_accuracy_{x_type}_{instances}.pdf", dpi=300, bbox_inches='tight', transparent=True)


# taken from https://stackoverflow.com/a/26369255
def save_all_figures(figs=None, extension="pdf", close=False):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:  # type
        fig.savefig("../{}.{}".format(fig._suptitle._text, extension))
        if close:
            plt.close(fig)
