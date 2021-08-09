import matplotlib.pyplot as plt

# verbosity
VERBOSE_MODEL_TRAINING = False  # print debug output of training the model
N_PRINT_WARNINGS = 49  # number of monitor warnings printed
N_PRINT_NOVELTIES = 49  # number of monitor novelties printed
PRINT_CONVEX_HULL_SAVED_VERTICES = False  # print number of vertices saved by convex hulls
PRINT_CREDIBILITY = True  # print credibility of abstractions


# external files
def readline_and_remove_line_break(file):
    result = file.readline()
    if result[-1] == "\n":
        result = result.split("\n")[0]
    return result


config_file = open("../paths.txt", "r")
MODEL_PATH = readline_and_remove_line_break(config_file)
DATA_PATH = readline_and_remove_line_break(config_file)
MODEL_INSTANCE_PATH = MODEL_PATH + "instances/"

# plotting
PLOT_ADDITIONAL_FEEDBACK = False  # create additional plots for feedback?
N_HISTOGRAM_PLOTS_UPPER_BOUND = 0  # maximum number of histogram plots
PLOT_MEAN = False  # plot mean of abstractions? (requires COMPUTE_MEAN == True)
PLOT_NON_EPSILON_SETS = False  # plot inner sets if epsilon is used?
_PLOT_MONITOR_TRAINING_AXIS = None
_PLOT_MONITOR_RATES_AXIS = None
PLOT_MONITOR_PERFORMANCE = True  # plot monitor performance (pie charts)?
NOVELTY_MARKER = "*"  # marker for plotting novelties
ADDITIONAL_MARKER = "X"  # marker for plotting additional points
PLOT_CLUSTER_CENTER_SIZE = 200
COLOR_blue = [0, .4, .6]
COLOR_yellow = [1, 0.65, 0.25]
COLOR_red = [1, 0, 0]


def PLOT_MONITOR_TRAINING_AXIS():  # plot window for monitor training
    global _PLOT_MONITOR_TRAINING_AXIS
    if _PLOT_MONITOR_TRAINING_AXIS is None:
        _PLOT_MONITOR_TRAINING_AXIS = plt.figure().add_subplot()
    return _PLOT_MONITOR_TRAINING_AXIS


def PLOT_MONITOR_RATES_AXIS():
    global _PLOT_MONITOR_RATES_AXIS
    if _PLOT_MONITOR_RATES_AXIS is None:
        _PLOT_MONITOR_RATES_AXIS = plt.subplots(1, 3)
    return _PLOT_MONITOR_RATES_AXIS


# general
FILTER_ZERO_DIMENSIONS = False  # filter out dimensions that were zero in the monitor training?
REPORT_REJECTIONS = False  # report inputs that are rejected (and also plot them)?


# monitors
USE_EPSILON_RELATIVE = False  # bloating of the abstraction size
# True: multiply epsilon value with size; False: add epsilon value to size
MONITOR_MANAGER_MODE_NORMAL = 1  # normal mode for monitor manager: extract layer values and pass to all monitors
MONITOR_MANAGER_MODE_ALPHA = 2  # special mode for monitor manager: pass inputs to monitor
MONITOR_MANAGER_MODE_PASSIVE = 3  # special mode for monitor manager: only obtain predictions for inputs


# monitor-training related
COMPUTE_MEAN = False  # compute the mean of 'SetBasedAbstraction's?
CONVEX_HULL_REDUNDANCY_REMOVAL = False  # remove redundant vertices from convex hulls?
CONVEX_HULL_REMOVE_BATCHES = False  # remove several points at once in the convex hulls? (almost no practical effect)
ONLY_LEARN_FROM_CORRECT_CLASSIFICATIONS = True  # ignore misclassification samples during monitor training?
MONITOR_TRAINING_CONVERGENCE_RANGE = 0.001  # range that is considered "no change" during monitor training
MONITOR_TRAINING_WINDOW_SIZE = 5  # convergence window (= number of data points in which no change occurs) during
#                                   monitor training
PRINT_FLAT_CONVEX_HULL_WARNING = True  # print warning about flat convex hulls


def print_flat_convex_hull_warning():
    global PRINT_FLAT_CONVEX_HULL_WARNING
    if PRINT_FLAT_CONVEX_HULL_WARNING:
        PRINT_FLAT_CONVEX_HULL_WARNING = False
        print("Warning: Convex hull is flat, for which conversion to H-representation is not available.")


# monitor-running related
PROPOSE_CLASS = False  # let the monitor propose a class based on the mean? (requires COMPUTE_MEAN == True)
MAXIMUM_CONFIDENCE = 1.0  # maximum confidence for rejection
ACCEPTANCE_CONFIDENCE = 0.0  # confidence when accepting
INCREDIBLE_CONFIDENCE = 1.0  # confidence when rejecting due to incredibility
SKIPPED_CONFIDENCE_NOVELTY_MODE = -1.0  # confidence when training novelties (has no meaning)
SKIPPED_CONFIDENCE = 1.0  # confidence when no distance is used
CONVEX_HULL_HALF_SPACE_DISTANCE_CORNER_CASE = 0.0  # half-space confidence for flat convex hulls
COMPOSITE_ABSTRACTION_POLICY = 2  # policy for CompositeAbstraction and multi-layer monitors; possible values:
#                                   1: average
#                                   2: maximum
SKIP_CONFIDENCE_COMPUTATION = True # skip computation of confidence (i.e., distance)?


# dimension reduction
REDUCTION_METHOD = "PCA"
MAX_N_COMPONENTS = 10

# box confidence; distribution fitting
DISTRIBUTION_METHOD = "GMM"
GMM_CONFIDENCE_TRAIN = 0.0  # confidence required to consider a point for training the abstraction
GMM_CONFIDENCE_TEST = 0.9  # confidence required to consider a point inside the abstraction


# --- defaults --- #


# data-loading related
N_TRAIN = 2000  # number of training data points
N_TEST = 1000  # number of testing data points
N_RUN = 1000  # number of running data points
RANDOMIZE_DATA = False  # randomize the data after loading?
CLASSES = [0, 1]  # classes for filtering (empty list: any)
# network related
N_EPOCHS = 10  # number of training epochs
BATCH_SIZE = 128  # batch size
USE_TRANSFER_LEARNING = False  # True for online monitoring
# monitor related
DEFAULT_N_FALSE_WARNINGS_THRESHOLD = 10
INITIAL_DISTANCE_THRESHOLD = 1.0  # initial threshold for distance monitor
# online-loop related
STATUS_ONLINE = 0
STATUS_RETRAIN_NETWORK = 1
DATA_BATCH_SIZE = 128
AUTHORITY_THRESHOLD_PERCENTAGE = 0.05  # the authority's threshold is determined as this percentage of the average
                                       # initial class dataset size
# utilities
DEFAULT_SEED = 0  # default random seed
