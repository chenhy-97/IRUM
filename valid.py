import torch
from utils import confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def dice_coefficient(binary_mask_true, binary_mask_pred):
    """
    Calculate the Dice Coefficient for binary masks.

    Parameters:
    binary_mask_true (numpy.ndarray): The ground truth binary mask.
    binary_mask_pred (numpy.ndarray): The predicted binary mask.

    Returns:
    float: Dice coefficient.
    """
    # Calculate intersection and union
    intersection = np.sum(binary_mask_true * binary_mask_pred)
    union = np.sum(binary_mask_true) + np.sum(binary_mask_pred)

    # Dice coefficient
    dice = 2.0 * intersection / union if union != 0 else 1.0

    return dice

def cul_iou(binary_mask_true, binary_mask_pred):
    """
    Calculate the Intersection over Union (IoU) for binary masks.

    Parameters:
    binary_mask_true (numpy.ndarray): The ground truth binary mask.
    binary_mask_pred (numpy.ndarray): The predicted binary mask.

    Returns:
    float: Intersection over Union.
    """
    # Calculate intersection and union
    intersection = np.sum((binary_mask_true == 1) & (binary_mask_pred == 1))
    union = np.sum((binary_mask_true == 1) | (binary_mask_pred == 1))

    # IoU
    iou_score = intersection / union if union != 0 else 1.0

    return iou_score

def determine_foreground_background(feature_map1, feature_map2):
    """
    Determine foreground and background pixels based on the maximum probability
    from two feature maps.

    Parameters:
    feature_map1 (numpy.ndarray): The first feature map (e.g., probability of being foreground).
    feature_map2 (numpy.ndarray): The second feature map (e.g., probability of being background).

    Returns:
    numpy.ndarray: A binary mask where 1 represents foreground and 0 represents background.
    """

    # Compare the two feature maps. If feature_map1 has a higher value, mark as foreground (1), else background (0)
    return np.where(feature_map1 > feature_map2, 0, 1)

rate = 38 * 5


def calculate_metrics(confusion_matrix):
    # 按照二分类的标准，我们取第一个元素为TP，最后一个为TN
    TP = confusion_matrix[1, 1]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[0, 0]

    sensitivity = TP / (TP + FN)  # 敏感性（召回率）
    specificity = TN / (TN + FP)  # 特异性
    accuracy = (TP + TN) / np.sum(confusion_matrix)  # 准确率

    # 精确率（Precision）和F1值
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # 防止除以0
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0  # 防止除以0

    return accuracy, sensitivity, specificity, precision, f1_score


def valid(config, net, val_loader, criterion):

    device = next(net.parameters()).device
    net.eval()

    print("START VALIDING")
    epoch_loss = 0
    y_true, y_score = [], []

    dice = []
    iou = []

    cm = torch.zeros((config.class_num, config.class_num))
    for i, pack in enumerate(val_loader):
        images = pack['imgs'].to(device)
        if images.shape[1] == 1:
            images = images.expand((-1, 3, -1, -1))
        names = pack['names']
        labels = pack['labels'].to(device)
        labels_segs = pack['region']
        output_j1, xd,yd, loss_f = net(images[:,0:3,:,:], images[:, 3:6, :, :])

        x_pre = determine_foreground_background(torch.squeeze(xd).cpu().numpy()[0],torch.squeeze(xd).cpu().numpy()[1])
        y_pre = determine_foreground_background(torch.squeeze(yd).cpu().numpy()[0], torch.squeeze(yd).cpu().numpy()[1])

        x_label = torch.squeeze(labels_segs).cpu().numpy()[1]
        y_label = torch.squeeze(labels_segs).cpu().numpy()[3]

        iou.append(cul_iou(x_pre,x_label))
        iou.append(cul_iou(y_pre, y_label))

        dice.append(dice_coefficient(x_pre, x_label))
        dice.append(dice_coefficient(y_pre, y_label))

        output = output_j1

        loss = criterion(output_j1, labels)

        pred = output.argmax(dim=1)
        y_true.append(torch.squeeze(labels).cpu().numpy())
        y_score.append(output[0].softmax(0)[1].item())

        cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        epoch_loss += loss.cpu()

    avg_epoch_loss = epoch_loss / len(val_loader)

    acc, sen, spe, pre, f1score = calculate_metrics(np.array(cm))
    # 绘制roc曲线
    auc = roc_auc_score(y_true, y_score)
    m_iou = np.mean(iou)
    m_dice = np.mean(dice)
    return [avg_epoch_loss, acc, sen, spe, auc, pre, f1score, m_iou, m_dice], cm