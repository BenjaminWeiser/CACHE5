import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score , confusion_matrix


def compute_accuracy(y_test, y_pred):
    threshold = 0.5
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_class = (y_pred[:, 1] > threshold).astype(int)
    else:
        # then y pred class was already given
        y_pred_class = y_pred
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    active_pos_rate = tp / (tp + fn)
    inactive_neg_rate = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return active_pos_rate, inactive_neg_rate, accuracy

def compute_thresholds(y_pred , y_test) :
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    accuracies = {'actives': [], 'inactives': []}
    for threshold in thresholds:
        y_pred_class = (y_pred[:, 1] > threshold).astype(int)
        active_pos_rate, inactive_neg_rate, accuracy = compute_accuracy(y_test, y_pred_class)
        accuracies['actives'].append(active_pos_rate)
        accuracies['inactives'].append(inactive_neg_rate)
        f1_scores.append(f1_score(y_test, y_pred_class))

    # get threshold for 95% active accuracy
    for i in range(len(accuracies['actives'])):
        if accuracies['actives'][i] <= 0.95:
            threshold_95 = thresholds[i]
            break

    return thresholds, threshold_95, f1_scores


def display_optimal_stats(y_pred, y_test, thresholds, threshold_95, f1_scores,
                          plot=False):
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    y_pred_class = (y_pred[:, 1] > optimal_threshold).astype(int)
    active_pos_rate, inactive_neg_rate, accuracy = compute_accuracy(y_test, y_pred_class)
    print(f"Optimal threshold is: {optimal_threshold:.3f}, accuracy is: {accuracy:.3f}")
    print(f"Active rate at optimal threshold: {active_pos_rate:.3f}")
    print(f"Inactive rate at optimal threshold: {inactive_neg_rate:.3f}")

    y_pred_class = (y_pred[:, 1] > threshold_95).astype(int)
    active_acc_95, inactive_acc_95, accuracy_95 = compute_accuracy(y_test, y_pred_class)
    print(f"95% active threshold is: {threshold_95:.3f}, accuracy is: {accuracy_95:.3f}")
    print(f"Active rate at 95% active rate: {active_acc_95:.3f}")
    print(f"Inactive rate at 95% active rate: {inactive_acc_95:.3f}")

    if plot:
        # visualize the F1 score over all thresholds
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, f1_scores, label='F1 score')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title(
            f'Optimal Threshold Determination')
        plt.legend()
        plt.grid(True)
        plt.show()

        col_labels = ['Threshold', 'Accuracy', 'Active Rate', 'Inactive Rate']
        cell_text = [
            [f"{round(optimal_threshold, 2)}", f"{round(accuracy, 2)}", f"{round(active_pos_rate, 2)}",
             f"{round(inactive_neg_rate, 2)}"],
            [f"{round(threshold_95, 2)}", f"{round(accuracy_95, 2)}", f"{round(active_acc_95, 2)}",
             f"{round(inactive_acc_95, 2)}"]
        ]

        fig, ax = plt.subplots(1, 1)
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center')
        plt.show()
    return optimal_threshold, accuracy, active_pos_rate, inactive_neg_rate, threshold_95, accuracy_95, active_acc_95, inactive_acc_95


