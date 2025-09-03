import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_cm

def confusion_matrix_to_accuraccies(confusion_matrix):
    cn_acc_dict = {}
    if not (np.any(confusion_matrix)):
        cn_acc_dict["acc"] = 0
        cn_acc_dict["kappa"] = 0
        cn_acc_dict["precision"] = np.zeros((confusion_matrix.shape[0], 1))
        cn_acc_dict["recall"] = np.zeros((confusion_matrix.shape[0], 1))
        cn_acc_dict["f1"] = np.zeros((confusion_matrix.shape[0], 1))
        cn_acc_dict["iou"] = np.zeros((confusion_matrix.shape[0], 1))
        cn_acc_dict["miou"] = 0
        cn_acc_dict["mf1"] = 0
        cn_acc_dict["cm"] = confusion_matrix

    else:
        confusion_matrix = confusion_matrix.astype(float)
        # sum(0) <- predicted sum(1) ground truth

        total = np.sum(confusion_matrix)
        n_classes, _ = confusion_matrix.shape
        overall_accuracy = np.sum(np.diag(confusion_matrix)) / (total + 1e-12)

        # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
        N = total
        p0 = np.sum(np.diag(confusion_matrix)) / N
        pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
        kappa = (p0 - pc) / (1 - pc + 1e-12)

        recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
        precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
        f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

        # iou
        true_positive = np.diag(confusion_matrix)
        false_positive = np.sum(confusion_matrix, 0) - true_positive
        false_negative = np.sum(confusion_matrix, 1) - true_positive
        iou = true_positive / (true_positive + false_positive + false_negative + 1e-12)
        miou = np.nanmean(iou)
        mf1=np.nanmean(f1)
        
        cn_acc_dict["acc"] = overall_accuracy
        cn_acc_dict["kappa"] = kappa
        cn_acc_dict["precision"] = precision
        cn_acc_dict["recall"] = recall
        cn_acc_dict["f1"] = f1
        cn_acc_dict["iou"] = iou
        cn_acc_dict["mf1"] = mf1
        cn_acc_dict["miou"] = miou
        cn_acc_dict["cm"] = confusion_matrix
    return cn_acc_dict


def build_confusion_matrix(targets, predictions, num_class,ignore_index):
    # labels = np.unique(targets)
    # labels = labels.tolist()
    # nclasses = len(labels)
    labels = np.arange(0, num_class)
    unique_label = np.unique(targets)
    if len(unique_label) == 1 and unique_label[0] == ignore_index:
        cm = np.zeros((num_class, num_class))
    else:
        cm = sklearn_cm(targets, predictions, labels=labels)
    #    precision = precision_score(targets, predictions, labels=labels, average='macro')
    #    recall = recall_score(targets, predictions, labels=labels, average='macro')
    #    f1 = f1_score(targets, predictions, labels=labels, average='macro')
    #    kappa = cohen_kappa_score(targets, predictions, labels=labels)
    # print('precision, recall, f1, kappa: ', precision, recall, f1, kappa)

    return cm
