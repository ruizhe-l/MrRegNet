import numpy as np
import sklearn.metrics as skm

# evaluation -----------------------------------------------
def softmax(prob_map, axis=1):
    e = np.exp(prob_map - np.max(prob_map))
    return e / np.sum(e, axis, keepdims=True)
    
def cross_entropy(pred, gt, epsilon=1e-9):
    axis = tuple(range(np.ndim(pred) - 1))# if np.ndim(pred) > 1 else -1
    ce = -np.sum(gt * np.log(pred + epsilon), axis) / pred.shape[0]
    return ce

def mean_squared_error(pred, gt):
    if np.ndim(gt) < 2:
        gt = np.expand_dims(gt, -1) 
    mse = np.mean(np.square(pred - gt), -1)
    return mse

def true_positive(pred, gt):
    axis = tuple(range(2, np.ndim(pred)))
    return (pred & gt).sum(axis)

def true_negative(pred, gt):
    axis = tuple(range(2, np.ndim(pred)))
    return ((~pred) & (~gt)).sum(axis)

def false_positive(pred, gt):
    axis = tuple(range(2, np.ndim(pred)))
    return (pred & (~gt)).sum(axis)

def false_negative(pred, gt):
    axis = tuple(range(2, np.ndim(pred)))
    return ((~pred) & gt).sum(axis)

def precision(pred, gt, epsilon=1e-9):
    tp = true_positive(pred, gt)
    fp = false_positive(pred, gt)
    return tp / (tp + fp + epsilon)

def recall(pred, gt, epsilon=1e-9):
    tp = true_positive(pred, gt)
    fn = false_negative(pred, gt)
    return tp / (tp + fn + epsilon)

def sensitivity(pred, gt, epsilon=1e-9):
    return recall(pred, gt, epsilon)

def specificity(pred, gt, epsilon=1e-9):
    tn = true_negative(pred, gt)
    fp = false_positive(pred, gt)
    return tn / (tn + fp + epsilon)

def accuracy(pred, gt):
    """ equal(pred, gt) / all(pred, gt)
        (tp + tn) / (tp + tn + fp + fn)
    """
    axis = tuple(range(1, pred.ndim))
    return (pred == gt).mean(axis)

def dice_coefficient(pred, gt, epsilon=1e-9, ignore_background=False, axis=None):
    """ 2 * intersection(pred, gt) / (pred + gt) 
        2 * tp / (2*tp + fp + fn)
    """
    axis = tuple(range(2, pred.ndim)) if axis is None else axis
    intersection = (pred * gt).sum(axis)
    sum_ = (pred + gt).sum(axis)
    dice = 2 * intersection / (sum_ + epsilon)
    if ignore_background:
        dice = dice[:, 1:]
    return dice

def iou(pred, gt, epsilon=1e-9):
    """ intersection(pred, gt) / union(pred, gt)
        tp / (tp + fp + fn)
    """
    axis = tuple(range(2, pred.ndim))
    intersection = (pred * gt).sum(axis)
    union = (pred + gt).sum(axis) - intersection
    return intersection / (union + epsilon)


def gncc(x, y):
    axis = tuple(range(1, len(x.shape)))
    x_norm = (x - x.mean(axis, keepdim=True))
    y_norm = (y - y.mean(axis, keepdim=True))
    
    # Calculate the GNCC
    gncc = (x_norm * y_norm).sum(axis) / (x_norm[0].numel() * x.std(axis) * y.std(axis))

    return gncc

# ----------------------------------------------------------

