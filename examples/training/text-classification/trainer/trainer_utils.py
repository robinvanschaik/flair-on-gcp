import pandas as pd
import numpy as np
from sklearn.utils import class_weight


def create_weight_dict(delimiter, label_index):
    """
    Creates a dictionary of the label and weights in order to deal with
    imbalanced datasets. E.g. predicting an infrequent label correctly will
    result in a larger decrease in the loss than predicting a frequent class.
    Params:
        delimiter: the type of delimiter used to separate the .csv columns.
        label_index: the column index of the label column.
    Returns:
        class_weights_dict: a dictionary with the label name and weight to be
                           passed to the Flair trainer.
    """

    training_dataset = pd.read_csv("/root/text-classification/data/train.csv", delimiter=delimiter)

    unique_labels = np.unique(training_dataset.iloc[:, label_index])
    labels = training_dataset.iloc[:, label_index]

    class_weights = class_weight.compute_class_weight('balanced',
                                                      unique_labels,
                                                      labels)
    class_weights_dict = {}
    for unique_labels, class_weights in zip(unique_labels, class_weights):
        class_weights_dict[unique_labels] = class_weights

    return class_weights_dict
