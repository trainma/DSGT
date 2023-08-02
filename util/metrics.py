import numpy as np
import torch


def masked_mse(preds, labels, null_val=0.0):
    """
    > If the label is not equal to the null value, then the loss is the squared difference between the
    prediction and the label. If the label is equal to the null value, then the loss is zero
    
    :param preds: the predictions of the model
    :param labels: the actual values of the target variable
    :param null_val: the value that is considered to be a null value
    :return: The mean squared error of the predictions and labels.
    """
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.double()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    """
    It takes in the predictions and the labels and returns the RMSE of the predictions and labels
    
    :param preds: The predictions
    :param labels: The ground truth values
    :param null_val: The value that is considered missing
    :return: The masked_rmse function returns the square root of the masked_mse function.
    """
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=0.0):
    """
    > If the label is not equal to the null value, then the mask is 1.0, otherwise it is 0.0
    
    :param preds: the predictions of the model
    :param labels: the actual values of the target variable
    :param null_val: the value that is considered to be a null value
    :return: The mean absolute error of the predictions and labels.
    """
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.double()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0.0):
    """
    If the label is not null, then the loss is the absolute percentage error. If the label is null, then
    the loss is zero
    
    :param preds: the predictions
    :param labels: the actual values
    :param null_val: the value that is considered as a null value
    :return: The mean absolute percentage error (MAPE)
    """
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.double()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_huber(preds, labels, null_val=0.0):
    """
    If the absolute difference between the prediction and the label is less than 1, then the loss is the
    square of the difference. Otherwise, the loss is the difference minus 0.5
    
    :param preds: the predictions of the model
    :param labels: the actual values of the target variable
    :param null_val: the value that is considered to be a null value
    :return: The mean of the loss
    """
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.double()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    t = torch.abs(preds - labels)
    loss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_r2_score(preds, labels, null_val=np.nan):
    """
    Calculate the R2 score where the labels are not null

    :param preds: the predictions of the model
    :param labels: the actual values of the target variable
    :param null_val: the value that is considered to be a null value
    :return: The R2 score of the predictions and labels
    """
    mask = ~torch.isnan(labels) if np.isnan(null_val) else (labels != null_val)
    mask = mask.float()

    preds = preds[mask]
    labels = labels[mask]

    ssr = torch.sum((labels - preds) ** 2)
    sst = torch.sum((labels - torch.mean(labels)) ** 2)
    r2_score = 1 - (ssr / sst)

    return r2_score


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    r2_score = masked_r2_score(pred, real, 0.0).item()

    return mae, mape, rmse, r2_score
