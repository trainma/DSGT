import ast
import configparser
import time

import pandas as pd

import utils
from models import SGT
from models.SGT import DSGT
from util.AirDataset import Air_dataset
from util.AirDataset import read_poi_embedding
from utils import *

DATASET = "Beijing"  # BeijingAir or TianjingAir
# config_file = f"./{DATASET}.conf"
config_file = "/root/pro/Beijing.conf"
config = configparser.ConfigParser()
config.read(config_file)
parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--no_cuda", action="store_true", help="没有GPU")
parser.add_argument("--batch_size",
                    type=int,
                    default=config["model"]["batch_size"],
                    help="batch大小")
parser.add_argument("--num_of_vertices", type=int, default=config["model"]["num_of_vertices"],
                    help="传感器数量")
parser.add_argument("--in_dim", type=int, default=config["model"]["in_dim"],
                    help="输入维度")
parser.add_argument("--hidden_dims", type=list, default=ast.literal_eval(config["model"]["hidden_dims"]),
                    help="中间各STSGCL层的卷积操作维度", )
parser.add_argument("--first_layer_embedding_size", type=int, default=config["model"]["first_layer_embedding_size"],
                    help="第一层输入层的维度", )
parser.add_argument("--out_layer_dim", type=int,
                    default=config["model"]["out_layer_dim"], help="输出模块中间层维度", )
parser.add_argument("--history", type=int, default=config["model"]["history"],
                    help="每个样本输入的离散时序")
parser.add_argument("--horizon", type=int, default=config["model"]["horizon"],
                    help="每个样本输出的离散时序")
parser.add_argument("--strides", type=int, default=config["model"]["strides"],
                    help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3", )
parser.add_argument("--temporal_emb", type=eval,
                    default=config["model"]["temporal_emb"], help="是否使用时间嵌入向量", )
parser.add_argument("--spatial_emb", type=eval, default=config["model"]["spatial_emb"],
                    help="是否使用空间嵌入向量", )
parser.add_argument("--use_mask", type=eval, default=config["model"]["use_mask"],
                    help="是否使用mask矩阵优化adj")
parser.add_argument("--activation", type=str, default=config["model"]["activation"],
                    help="激活函数 {relu, GlU}", )
parser.add_argument("--seed", type=int, default=config["train"]["seed"],
                    help="种子设置")
parser.add_argument("--learning_rate", type=float, default=config["train"]["learning_rate"],
                    help="初始学习率", )
parser.add_argument("--lr_decay", type=eval, default=config["train"]["lr_decay"],
                    help="是否开启初始学习率衰减策略")
parser.add_argument("--lr_decay_step", type=str, default=config["train"]["lr_decay_step"],
                    help="在几个epoch进行初始学习率衰减", )
parser.add_argument("--lr_decay_rate", type=float,
                    default=config["train"]["lr_decay_rate"], help="学习率衰减率", )
parser.add_argument("--epochs", type=int, default=config["train"]["epochs"],
                    help="训练代数")
parser.add_argument("--print_every", type=int, default=config["train"]["print_every"],
                    help="几个batch报训练损失", )
parser.add_argument("--save", type=str, default=config["train"]["save"],
                    help="保存路径")
parser.add_argument("--expid", type=int, default=config["train"]["expid"],
                    help="实验 id")
parser.add_argument("--max_grad_norm", type=float, default=config["train"]["max_grad_norm"],
                    help="梯度阈值")
parser.add_argument("--patience", type=int, default=config["train"]["patience"],
                    help="等待代数")
parser.add_argument("--log_file", default=config["train"]["log_file"],
                    help="log file")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dataset", type=str, default=DATASET)
parser.add_argument("--layers", type=int, default=config["train"]["layers"])
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def exp_test(checkpoint_path):
    if DATASET == "Beijing":
        poi_file = './data/BeiJingAir/poi_embed2.txt'
        spatial_adj = pd.read_csv(
            './data/BeiJingAir/spatial_corr.csv', header=None).values
    elif DATASET == "Tianjin":
        poi_file = './data/TianJingAir/poiEmbedding2.txt'
        spatial_adj = pd.read_csv(
            './data/TianJingAir/spatial_graph.csv').values

    # poi_file = './data/TianJingAir/poiEmbedding2.txt'
    # spatial_adj = pd.read_csv('./data/TianJingAir/spatial_graph.csv').values
    # poi_adj = pd.read_csv('./data/processed/Air/                        poi_corr.csv').values
    # temporal_adj = pd.read_csv(
    #     './data/processed/Air/temporal_corr2.csv').values
    spatial_adj_local = torch.FloatTensor(construct_adj_local(spatial_adj, 3))
    spatial_adj_1hop = torch.FloatTensor(
        construct_dynamic_adj_1hop(spatial_adj, 3))
    spatial_adj_2hop = torch.FloatTensor(
        construct_dynamic_adj_lt(spatial_adj, 3))

    adj_list = [spatial_adj_local, spatial_adj_1hop, spatial_adj_2hop]
    test_dataset = Air_dataset(data=args.dataset,
                               flag='test',
                               seq_len=args.history,
                               pred_len=args.horizon)
    test_loader = data.DataLoader(test_dataset,
                                  args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    n_layers = args.layers
    d_model = 64
    dropout = 0.3
    horizon = args.horizon
    num_of_vertices = args.num_of_vertices
    seq_len = args.history
    pre_graph = torch.from_numpy(spatial_adj).float().to(device)
    model = DSGT(batch_size=args.batch_size, adj_list=adj_list, n_layers=n_layers, d=8, K=8,
                 seq_len=seq_len, bn_decay=0.1,
                 nodes=num_of_vertices, d_model=d_model, dropout=dropout, pred_len=horizon, Satt=True, Tatt=True, pre_graph=pre_graph).to(
        device)
    model.load_state_dict(torch.load(checkpoint_path))

    print('load model from: ', checkpoint_path)
    test_loss = []
    test_mape = []
    test_rmse = []
    test_mae_station = []
    test_rmse_station = []
    outputs = []
    real = []
    model.eval()
    for x, y, x_mark, y_mark, x_meter in test_loader:
        PoiE = read_poi_embedding(poi_file)
        PoiE = PoiE.to(torch.float32).to(device)
        batch_x = x.to(torch.float32).to(device)
        batch_y = y[:, :, :, 0].to(torch.float32).to(device)
        batch_x_mark = x_mark.to(torch.float32).to(device)
        batch_y_mark = y_mark.to(torch.float32).to(device)
        batch_x_meter = x_meter.to(torch.float32).to(device)

        output = model(batch_x, batch_x_mark, batch_y_mark,
                       batch_x_meter, PoiE)
        predict = test_dataset.scaler.inverse_transform(output)  # B, T, N
        real_val = y
        # real_val = y.unsqueeze(-1)

        predict = predict.cpu().detach()
        mae = utils.masked_mae(predict, real_val, 0.0).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()
        mae_station = []
        rmse_station = []
        for i in range(predict.shape[2]):
            mae_station.append(utils.masked_mae(
                predict[:, :, i, :].detach().cpu(), real_val[:, :, i, :].detach().cpu()))
            rmse_station.append(utils.masked_rmse(
                predict[:, :, i, :].detach().cpu(), real_val[:, :, i, :].detach().cpu()))

        test_loss.append(mae)
        test_mape.append(mape)
        test_rmse.append(rmse)
        test_mae_station.append(mae_station)
        test_rmse_station.append(rmse_station)
        outputs.append(predict)
        real.append(real_val)
    y_hat = torch.cat(outputs, dim=0)
    y_real = torch.cat(real, dim=0)
    y_hat = y_hat.numpy()
    y_real = y_real.numpy()
    np.save(f'./yhat_{horizon}_{DATASET}.npy', y_hat)
    np.save(f'./yreal_{horizon}_{DATASET}.npy', y_real)
    print(f'./yhat_{horizon}_{DATASET}.npy' + ' have save done')
    print(f'./yreal_{horizon}_{DATASET}.npy' + ' have save done')

    test_mae_station = np.array(test_mae_station)
    test_rmse_station = np.array(test_rmse_station)
    test_mae_station = np.mean(test_mae_station, axis=0)
    test_rmse_station = np.mean(test_rmse_station, axis=0)
    mtest_loss = np.mean(test_loss)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)
    test_logs = "Test Loss: {:.4f} Test MAPE: {:.4f}, Test RMSE: {:.4f}".format(
        mtest_loss, mtest_mape, mtest_rmse)
    print(test_logs)


if __name__ == "__main__":
    start = time.time()
    exp_test('./garage/Air/12/exp_12_12_17.36_Beijing_best_model.pth')
    end = time.time()
