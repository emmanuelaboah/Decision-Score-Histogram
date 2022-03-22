import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def save_plot(fig, path="plots", tight_layout=True, fig_extension="png", resolution=400):
    """
        Function for saving figures and plots
        :arg
            1. fig: label of the figure
            2. path (optional): output path of the figure
    """
    img_path = os.path.join(".", path)
    os.makedirs(img_path, exist_ok=True)
    fig_path = os.path.join(img_path, fig + "." + fig_extension)

    print("Saving figure...", fig)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)
    print("figure can be found in: ", path)


class AnomalyScoreHist:
    """
        Histogram based visualization approach for
        anomaly detection algorithms and supervised classification
        algorithms involving binary predictions.
    """

    def check_input_dtype(self, input_data):
        if isinstance(input_data, np.ndarray):
            return input_data.reshape(-1, 1)

        elif isinstance(input_data, (pd.DataFrame, pd.Series)):
            return input_data.to_numpy().reshape(-1, 1)

        else:
            try:
                data = np.array(input_data)
            except:
                raise Exception('Wrong data type')
            else:
                return data.reshape(-1, 1)

    def check_gtruth_dtype(self, ground_truth):
        g_truth = self.check_input_dtype(ground_truth)
        warning_msg = 'ground truth must be an array of 1s (positive scores) and -1s (negative scores)'
        assert all(label in [1, -1] for label in g_truth), warning_msg

        return g_truth

    def compute_hist_data(self, dec_score, ground_truth):
        dec_score = self.check_input_dtype(dec_score)
        ground_truth = self.check_gtruth_dtype(ground_truth)

        false_positives = []
        false_negatives = []
        for idx in range(len(dec_score)):
            if ground_truth[idx] == -1 and dec_score[idx] >= 0:
                false_positives.append(dec_score[idx])

            elif ground_truth[idx] == 1 and dec_score[idx] < 0:
                false_negatives.append(dec_score[idx])

            else:
                continue

        true_positives = []
        true_negatives = []
        for idx in range(len(dec_score)):
            if ground_truth[idx] == 1 and dec_score[idx] >= 0:
                true_positives.append(dec_score[idx])

            elif ground_truth[idx] == -1 and dec_score[idx] < 0:
                true_negatives.append(dec_score[idx])

            else:
                continue

        all_positives = [dec_score[idx] for idx in range(len(dec_score))
                         if dec_score[idx] >= 0]
        all_negatives = [dec_score[idx] for idx in range(len(dec_score))
                         if dec_score[idx] < 0]

        pos_scaler = MinMaxScaler().fit(all_positives)
        neg_scaler = MinMaxScaler().fit(all_negatives)

        if len(true_positives) == 0:
            true_positives = []
        else:
            true_positives = np.round(pos_scaler.transform(true_positives), 3)

        if len(false_positives) == 0:
            false_positives = []
        else:
            false_positives = np.round(pos_scaler.transform(false_positives), 3)

        if len(true_negatives) == 0:
            true_negatives = []
        else:
            true_negatives = np.round(neg_scaler.transform(true_negatives) * -1, 3)

        if len(false_negatives) == 0:
            false_negatives = []
        else:
            false_negatives = np.round(neg_scaler.transform(false_negatives) * -1, 3)

        return true_positives, true_negatives, false_positives, false_negatives

    def plot_hist(self, dec_score, ground_truth, fig_name='hist_plot'):
        TP, TN, FP, FN = self.compute_hist_data(dec_score, ground_truth)

        plt.figure(figsize=(9, 7))
        plt.hist(TP, weights=np.ones(len(TP)) / (len(TP) + len(FN)), facecolor='C0',
                 bins=50, label='True Positive', alpha=0.5)
        plt.hist(TN, weights=np.ones(len(TN)) / (len(TN) + len(FP)), facecolor='C1',
                 bins=50, label='True Negative', alpha=0.5)
        plt.hist(FP, weights=np.ones(len(FP)) / (len(TN) + len(FP)), facecolor='C1',
                 bins=50, label='False Positive')
        plt.hist(FN, weights=np.ones(len(FN)) / (len(TP) + len(FN)), facecolor='C0',
                 bins=50, label='False Negative')

        plt.xlabel('Decision Score', fontsize=27, labelpad=10)
        plt.ylabel('% Prediction', fontsize=27, labelpad=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlim([-1.1, 1.1])
        plt.ylim([0, 1.05])
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.axvline(x=0, linestyle='--', linewidth=1, color='black')
        plt.legend(loc='best', fontsize=18)
        save_plot(fig_name)

        plt.show()

