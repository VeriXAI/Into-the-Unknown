import numpy as np

from utils import rejected_inputs, DataSpec, get_model, obtain_predictions, reduce_dimension, plot_2d_projection, \
    CsvStorage, store_csv_storage


def plot_and_store_rejections(model_name, model_path, data_name, data_run, history_run, monitor_manager,
                              layer, class_label_map, all_classes, dimensions, confidence_threshold,
                              group_consecutive=-1, row_header=None):
    # group_consecutive: nonpositive numbers means ignored
    #                    positive number means to use a different color for consecutive indices of at least the given
    #                    length; examples for index list [2, 3, 7, 8, 9, 12]:
    #                    <= 0: [[2, 3, 7, 8, 9, 12]]  (1 color, all indices)
    #                    1: [[2, 3], [7, 8, 9], [12]]  (3 colors, all indices)
    #                    2: [[2, 3], [7, 8, 9]]  (2 colors)
    #                    3: [[7, 8, 9]]  (1 color)
    #                    4: []  (no index)
    # row_header: None if the indices shall be stored
    #             list of column names if the inputs shall be stored instead

    monitor2rejected_indices = rejected_inputs(history=history_run, confidence_threshold=confidence_threshold)
    for monitor in monitor_manager.monitors():
        m_id = monitor.id()
        rejected_indices = monitor2rejected_indices[m_id]
        x_rejected = data_run.inputs()[rejected_indices]
        y_rejected = data_run.ground_truths()[rejected_indices]
        data_run_rejected = DataSpec(inputs=x_rejected, labels=y_rejected)
        model, history_model = get_model(model_name=model_name, model_path=model_path)
        layer2values_rejected, _, _ = obtain_predictions(model=model, data=data_run_rejected,
                                                         class_label_map=class_label_map, layers=[layer])
        if monitor_manager.layer2n_components:
            layer2values_rejected, _ = reduce_dimension(layer2data=layer2values_rejected, layers=[layer],
                                                        layer2components=monitor_manager.layer2components,
                                                        layer2n_components=monitor_manager.layer2n_components,
                                                        method_name=monitor_manager.reduction_method)
        values_rejected = layer2values_rejected[layer]

        rejected_groups = []
        if group_consecutive <= 0:
            rejected_values_lists = [values_rejected]
        else:
            # extract consecutive sequences of indices of length at least 'paint_consecutive'
            rejected_values_lists = []
            if len(rejected_indices) > 0:
                current_group = 1
                i_start = 0
                v_prev = rejected_indices[0]
                i_next = 1
                while i_next < len(rejected_indices):
                    v_next = rejected_indices[i_next]
                    if v_next > v_prev + 1:
                        interval_length = i_next - i_start
                        if interval_length >= group_consecutive:
                            rejected_values_lists.append(values_rejected[i_start:i_next])
                            group_to_add = current_group
                            current_group += 1
                        else:
                            group_to_add = 0
                        for _ in range(interval_length):
                            rejected_groups.append(group_to_add)
                        i_start = i_next

                    i_next += 1
                    v_prev = v_next
                interval_length = i_next - i_start
                if interval_length >= group_consecutive:
                    rejected_values_lists.append(values_rejected[i_start:i_next])
                    group_to_add = current_group
                else:
                    group_to_add = 0
                for _ in range(interval_length):
                    rejected_groups.append(group_to_add)

        # plot rejections
        plot_2d_projection(history=history_run, monitor=monitor, layer=layer,
                           all_classes=all_classes,  class_label_map=class_label_map, category_title=model_name,
                           dimensions=dimensions, additional_point_lists=rejected_values_lists)

        # store rejection indices to file
        if row_header is None:
            # store indices
            row_header = ["index"]
            input_storage = np.array([[index] for index in rejected_indices])
        else:
            # store inputs instead of indices
            input_storage = x_rejected
        if group_consecutive:
            row_header.append("group")
            input_storage = np.c_[input_storage, np.array([[g] for g in rejected_groups])]
        csv_storage = CsvStorage(rows=input_storage, row_header=row_header)
        filename = "rejected_inputs_{}_monitor_{:d}.csv".format(data_name, m_id)
        store_csv_storage(filename, csv_storage)
