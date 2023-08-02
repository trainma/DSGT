import torch.optim as optim
from model import *
from models.SGT import SGT, DSGT
import utils
import numpy as np
import torch


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


class trainer():
    def __init__(self,
                 args,
                 scaler,
                 train_scaler,
                 val_scaler,
                 test_scaler,
                 adjList,
                 seq_len,
                 d_model,
                 n_layers,
                 num_of_vertices,
                 lr,
                 device,
                 log=None,
                 max_grad_norm=5,
                 lr_decay=False,
                 temporal_emb=True,
                 spatial_emb=True,
                 horizon=12,
                 strides=3,
                 dropout=0.3,
                 test=None,
                 pre_graph=None,
                 in_dim=1,
                 meter_flag=None ):
        super(trainer, self).__init__()
        # Creating a model.
        pre_graph = torch.from_numpy(pre_graph).float().to(device)
        self.model = DSGT(args.batch_size, adj_list=adjList, n_layers=n_layers, d=8, K=int(d_model / 8),
                          seq_len=seq_len,
                          bn_decay=0.1,
                          nodes=num_of_vertices, d_model=d_model, dropout=dropout, pred_len=horizon, Satt=True,
                          Tatt=True, pre_graph=pre_graph,meter_flag=meter_flag,in_dim=in_dim)
        self.model.to(device)
        # self.model_parameters_init()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            eps=1.0e-8,
            weight_decay=1e-4)

        if lr_decay:
            lr_decay_steps = [
                int(i) for i in list(args.lr_decay_step.split(','))
            ]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=lr_decay_steps,
                gamma=args.lr_decay_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.scaler = scaler
        self.train_scaler = train_scaler
        self.val_scaler = val_scaler
        self.test_scaler = test_scaler
        self.clip = max_grad_norm
        if log is not None:
            utils.log_string(
                log, "model trainable parameter: {:,}".format(utils.count_parameters(self.model)))
            utils.log_string(
                log,
                'GPU usage:{:,}'.format(torch.cuda.max_memory_allocated() /
                                      1000000 if torch.cuda.is_available() else 0))

    def model_parameters_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.0003)
            else:
                nn.init.uniform_(p)

    def train(self, input, input_mark, y_mark, input_meter, real_val, poiE):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_mark, y_mark,
                            input_meter, poiE)  # B, T, N, D
        predict = self.train_scaler.inverse_transform(output)  # B, T, N,1
        real_val = real_val.unsqueeze(-1)  # B, T, N, 1
        loss = self.loss(predict, real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = utils.masked_mae(predict, real_val).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()

        return loss.item(), mae, mape, rmse

    def evel(self, input, input_mark, y_mark, input_meter, real_val, poiE):
        self.model.eval()
        output = self.model(input, input_mark, y_mark,
                            input_meter, poiE)  # B, T, N
        predict = self.val_scaler.inverse_transform(output)  # B, T, N
        real_val = real_val.unsqueeze(-1)  # B, T, N, 1
        mae = utils.masked_mae(predict, real_val, 0.0).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()

        return mae, mape, rmse

    def test(self, input, input_mark, y_mark, input_meter, real_val, poiE):
        self.model.eval()
        output = self.model(input, input_mark, y_mark,
                            input_meter, poiE)  # B, T, N
        predict = self.test_scaler.inverse_transform(output)  # B, T, N
        real_val = real_val.unsqueeze(-1)  # B, T, N, 1

        mae = utils.masked_mae(predict, real_val, 0.0).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()

        return mae, mape, rmse
