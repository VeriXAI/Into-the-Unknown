from run.experiment_helper import *
from run_experiment_online import run_experiment_online


def run_experiment_online_monitor_comparison():
    print("---\nStarting benchmark for monitor comparison.\n---\n")

    # monitor choice
    monitor_options = [
        {"random": False, "alpha": 0.0, "distance": True},  # box distance
        {"random": False, "alpha": 0.0, "distance": False},  # box abstraction
        {"random": False, "alpha": 0.1, "distance": False},  # alpha threshold
        {"random": True, "alpha": 0.0, "distance": False},  # random rejection
    ]

    # benchmark instances with these entries: (instance, random probability, authority threshold percentage)
    # 'instance' itself is a tuple with these entries:
    # (instance constructor, box_abstraction, epsilon, clustering threshold, initial classes, model instance)
    benchmark_instances = [
        ((instance_MNIST, box_abstraction_MNIST, 0.0, 0.3, 5, 1), 0.95, 0.05, "PCA"),
        ((instance_F_MNIST, box_abstraction_F_MNIST, 0.0, 0.07, 5, 1), 0.95, 0.05, "PCA"),
        ((instance_CIFAR10, box_abstraction_CIFAR10, 0.0, 0.3, 5, 1), 0.95, 0.01, "KernelPCA"),
        ((instance_GTSRB, box_abstraction_GTSRB, 0.0, 0.3, 22, 1), 0.95, 0.05, "PCA"),
        ((instance_EMNIST, box_abstraction_EMNIST, 0.0, 0.3, 24, 1), 0.90, 0.05, "PCA")
    ]

    # indices of runs (with corresponding random seed) on the same instance
    run_range = range(1, 6)

    for monitor_option in monitor_options:
        do_reduce_dimension = True
        use_distance_monitor = monitor_option["distance"]
        alpha_threshold = monitor_option["alpha"]
        use_random_monitor = monitor_option["random"]

        for benchmark_instance, random_monitor_acceptance_probability, authority_threshold_percentage, reduction_method\
                in benchmark_instances:
            if use_random_monitor:
                random_monitor_acceptance_probability_final = random_monitor_acceptance_probability
            else:
                random_monitor_acceptance_probability_final = 0.0
            interaction_limit = 1 - random_monitor_acceptance_probability

            run_experiment_online(do_reduce_dimension=do_reduce_dimension, use_distance_monitor=use_distance_monitor,
                                  alpha_threshold=alpha_threshold,
                                  random_monitor_acceptance_probability=random_monitor_acceptance_probability_final,
                                  instances=[benchmark_instance], run_range=run_range,
                                  interaction_limit=interaction_limit,
                                  authority_threshold_percentage=authority_threshold_percentage,
                                  reduction_method=reduction_method)


def run_experiment_online_ablation_threshold():
    print("---\nStarting ablation study for distance threshold.\n---\n")

    # settings
    run_range = [1]
    interaction_limit = 0.05
    authority_threshold_percentage = 0.05
    reduction_method = "PCA"

    for i_model_instance in range(2, 7):
        initial_distance_threshold = INITIAL_DISTANCE_THRESHOLD
        do_reduce_dimension = True
        adapt_score_thresholds = True
        if i_model_instance == 3:
            # static distance threshold
            adapt_score_thresholds = False
        elif i_model_instance == 4:
            # dynamic distance threshold without PCA
            do_reduce_dimension = False
        elif i_model_instance == 5:
            # dynamic distance threshold with initial value 0.7
            initial_distance_threshold = 0.7
        elif i_model_instance == 6:
            # dynamic distance threshold with initial value 1.3
            initial_distance_threshold = 1.3

        # benchmark instance (only differ in model-instance number)
        instance = (instance_MNIST, box_abstraction_MNIST, 0.0, 0.3, 5, i_model_instance)

        run_experiment_online(instances=[instance], run_range=run_range, interaction_limit=interaction_limit,
                              use_distance_monitor=True, alpha_threshold=0.0, random_monitor_acceptance_probability=0.0,
                              authority_threshold_percentage=authority_threshold_percentage,
                              do_reduce_dimension = do_reduce_dimension, reduction_method = reduction_method,
                              initial_distance_threshold=initial_distance_threshold,
                              adapt_score_thresholds=adapt_score_thresholds)


def run_experiment_online_all():
    run_experiment_online_monitor_comparison()
    run_experiment_online_ablation_threshold()


if __name__ == "__main__":
    run_experiment_online_all()
