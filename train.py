import ast
import configparser
import datetime
import time

import tqdm
from engine import trainer
from util.AirDataset import Air_dataset, read_poi_embedding
from utils import *

DATASET = "Beijing"  # Beijing or Tianjin
config_file = f"./{DATASET}.conf"
config = configparser.ConfigParser()
config.read(config_file)
parser = argparse.ArgumentParser(description="arguments")
parser.add_argument("--dataset", type=str, default=DATASET)
parser.add_argument("--batch_size", type=int,
                    default=config["model"]["batch_size"], help="the size of batch")
parser.add_argument("--num_of_vertices", type=int,
                    default=config["model"]["num_of_nodes"], help="the num of node")
parser.add_argument("--in_dim", type=int, default=config["model"]["in_dim"],
                    help="The number of input feature")
parser.add_argument("--history", type=int, default=config["model"]["history"],
                    help="The number of time window")
parser.add_argument("--horizon", type=int, default=config["model"]["horizon"],
                    help="The number of predict time step")
parser.add_argument("--strides", type=int, default=config["model"]["strides"],
                    help="the number of slide window in MSTS-GCC", )
parser.add_argument("--temporal_emb", type=eval,
                    default=config["model"]["temporal_emb"], help="whether to use temporal embedding", )
parser.add_argument("--spatial_emb", type=eval, default=config["model"]["spatial_emb"],
                    help="whether to use spatial embedding", )
parser.add_argument("--use_mask", type=eval, default=config["model"]["use_mask"],
                    help="whether to use mask matrix")
parser.add_argument("--learning_rate", type=float, default=config["train"]["learning_rate"],
                    help="initial learning rate", )
parser.add_argument("--lr_decay", type=eval, default=config["train"]["lr_decay"],
                    help="whether to use lr decay", )
parser.add_argument("--lr_decay_step", type=str, default=config["train"]["lr_decay_step"],
                    help="the number of epoch to lr decay", )
parser.add_argument("--lr_decay_rate", type=float,
                    default=config["train"]["lr_decay_rate"], help="lr decay rate", )
parser.add_argument("--epochs", type=int, default=config["train"]["epochs"],
                    help="total train epochs")
parser.add_argument("--print_every", type=int, default=config["train"]["print_every"],
                    help="every ", )
parser.add_argument("--save", type=str, default=config["train"]["save"],
                    help="save path")
parser.add_argument("--expid", type=int, default=config["train"]["expid"],
                    help="exp id")
parser.add_argument("--patience", type=int, default=config["train"]["patience"],
                    help="early stop patience")
parser.add_argument("--log_file", default=config["train"]["log_file"],
                    help="log file")
parser.add_argument("--num_workers", type=int, default=config["train"]["num_workers"],
                    help="num workers")
parser.add_argument("--layers", type=int, default=config["train"]["layers"])
parser.add_argument("--d_model", type=int, default=config["model"]["d_model"])
parser.add_argument("--dropout", type=eval, default=config["train"]["dropout"])
parser.add_argument("--max_grad_norm", type=float, default=config["train"]["max_grad_norm"],
                    help="the max grad norm")
parser.add_argument("--meter_flag", type=eval, default=config["train"]["meter_flag"],
                    help="whether to use meter data in training process")
parser.add_argument("--weight_decay", type=eval, default=config["train"]["weight_decay"])
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
curr_time = datetime.datetime.now()
timestamp = datetime.datetime.strftime(curr_time, '%m-%d %H:%M')
log = open('log.txt', "a")
log_string(log, str(args))


def main() -> None:
    if DATASET == "Beijing":
        poi_file = './data/BeiJingAir/poi_embed2.txt'
        spatial_adj = pd.read_csv(
            './data/BeiJingAir/spatial_corr.csv', header=None).values
    elif DATASET == "Tianjin":
        poi_file = './data/TianJingAir/poiEmbedding2.txt'
        spatial_adj = pd.read_csv(
            './data/TianJingAir/spatial_graph.csv').values
    spatial_adj_local = torch.FloatTensor(construct_adj_local(spatial_adj, 3))
    spatial_adj_1hop = torch.FloatTensor(
        construct_dynamic_adj_1hop(spatial_adj, 3))
    spatial_adj_2hop = torch.FloatTensor(
        construct_dynamic_adj_lt(spatial_adj, 3))
    adj_list = [spatial_adj_local, spatial_adj_1hop, spatial_adj_2hop]

    test_dataset, test_loader, train_dataset, train_loader, val_dataset, valid_loader = initial_dataset()

    engine: trainer = trainer(
        args=args,
        scaler=None,
        train_scaler=train_dataset.scaler,
        val_scaler=val_dataset.scaler,
        test_scaler=test_dataset.scaler,
        adjList=adj_list,
        seq_len=args.history,
        d_model=args.d_model,
        n_layers=args.layers,
        num_of_vertices=args.num_of_vertices,
        log=log,
        lr=args.learning_rate,
        device=device,
        max_grad_norm=args.max_grad_norm,
        lr_decay=args.lr_decay,
        temporal_emb=args.temporal_emb,
        spatial_emb=args.spatial_emb,
        horizon=args.horizon,
        strides=args.strides,
        dropout=args.dropout,
        pre_graph=spatial_adj,
        in_dim=args.in_dim,
        meter_flag=args.meter_flag,
        weight_decay=args.weight_decay,

    )
    log_string(log, "Start training...")
    log_string(log, "=========================================")
    log_string(log, "compiling model...")
    his_loss = []
    val_time = []
    train_time = []
    wait = 0
    val_mae_min = float("inf")
    val_rmse_min = float("inf")
    for i in tqdm.tqdm(range(1, args.epochs + 1)):
        if wait >= args.patience:
            log_string(log, f"early stop at epoch: {i:04d}")
            break
        train_loss, train_mae, train_mape, train_rmse = [], [], [], []
        t1 = time.time()
        # dataloader["train_loader"].shuffle()
        for _, (x, y, x_mark, y_mark, x_meter) in enumerate(train_loader):

            poi_embedding = read_poi_embedding(poi_file)
            poi_embedding = poi_embedding.to(torch.float32).to(device)
            batch_x = x.to(torch.float32).to(device)
            batch_y = y[:, :, :, 0].to(torch.float32).to(device)
            batch_x_mark = x_mark.to(torch.float32).to(device)
            batch_y_mark = y_mark.to(torch.float32).to(device)
            batch_x_meter = x_meter.to(torch.float32).to(device)

            loss, tmae, tmape, trmse = engine.train(batch_x, batch_x_mark, batch_y_mark, batch_x_meter,
                                                    batch_y, poi_embedding)

            train_loss.append(loss)
            train_mae.append(tmae)
            train_mape.append(tmape)
            train_rmse.append(trmse)
            if _ % args.print_every == 0:
                logs = "Iter: {:03d}, Train Loss: {:.4f}, avg Train Loss: {:.4f}, lr: {}"
                print(
                    logs.format(
                        _,
                        train_loss[-1],
                        np.mean(train_loss),
                        engine.optimizer.param_groups[0]["lr"],
                    ),
                    flush=True,
                )

        if args.lr_decay:
            engine.lr_scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_mae, valid_mape, valid_rmse, valid_r2 = [], [], [], []
        test_loss, test_mape, test_rmse, test_r2 = [], [], [], []

        log_string(log, "Start validation...")
        s1: float = time.time()
        for _, (x, y, x_mark, y_mark, x_meter) in enumerate(valid_loader):
            poi_embedding = read_poi_embedding(poi_file)
            poi_embedding = poi_embedding.to(torch.float32).to(device)
            batch_x = x.to(torch.float32).to(device)
            batch_y = y[:, :, :, 0].to(torch.float32).to(device)
            batch_x_mark = x_mark.to(torch.float32).to(device)
            batch_y_mark = y_mark.to(torch.float32).to(device)
            batch_x_meter = x_meter.to(torch.float32).to(device)
            vmae, vmape, vrmse = engine.evel(batch_x, batch_x_mark, batch_y_mark,
                                             batch_x_meter, batch_y, poi_embedding)

            valid_mae.append(vmae)
            valid_mape.append(vmape)
            valid_rmse.append(vrmse)

        s2: float = time.time()
        logs = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        log_string(log, logs.format(i, (s2 - s1)))

        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mape = np.mean(valid_mape)
        mvalid_mae = np.mean(valid_mae)

        his_loss.append(mvalid_mae)

        logs = "Epoch: {:03d}, Train Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch"

        log_string(
            log,
            logs.format(
                i,
                mtrain_loss,
                mvalid_mae,
                mvalid_mape,
                mvalid_rmse,
                (t2 - t1),
            ),
        )
        save_path_ckpt = os.path.join(args.save, str(args.horizon))
        if not os.path.exists(save_path_ckpt):
            os.makedirs(save_path_ckpt)
        if mvalid_rmse <= val_mae_min:
            log_string(
                log,
                f"val rmse decrease from {val_mae_min:.4f} to {mvalid_rmse:.4f}, "
                f'save model to {args.save + "exp_" + str(args.expid) + "_" + str(round(mvalid_mae, 2)) + "_" + str(round(mvalid_rmse, 2)) + "_best_model.pth"}',
            )
            wait = 0
            val_mae_min = mvalid_rmse
            best_model_wts = engine.model.state_dict()
            torch.save(
                best_model_wts,
                save_path_ckpt + '/' + "exp_" + str(args.history) + "_" + str(args.horizon) + "_" +
                str(round(val_mae_min, 2)) + "_" + DATASET + "_best_model.pth",
            )
        else:
            wait += 1

        np.save("./history_loss" + f"_{args.expid}", his_loss)

        for _, (x, y, x_mark, y_mark, x_meter) in enumerate(test_loader):
            poi_embedding = read_poi_embedding(poi_file)
            poi_embedding = poi_embedding.to(torch.float32).to(device)
            batch_x = x.to(torch.float32).to(device)

            batch_y = y[:, :, :, 0].to(torch.float32).to(device)
            batch_x_mark = x_mark.to(torch.float32).to(device)
            batch_y_mark = y_mark.to(torch.float32).to(device)
            batch_x_meter = x_meter.to(torch.float32).to(device)
            # time.sleep(0.003)
            # vmae, vmape, vrmse = engine.evel(
            #     valx, val_mark, val_graph, val_meter, valy)
            vmae, vmape, vrmse = engine.test(batch_x, batch_x_mark, batch_y_mark,
                                             batch_x_meter, batch_y, poi_embedding)
            test_loss.append(vmae)
            test_mape.append(vmape)
            test_rmse.append(vrmse)

        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_rmse = np.mean(test_rmse)
        test_logs = "Epoch: {:03d}, \
                    Test MAE: {:.4f} Test MAPE: {:.4f}, Test RMSE: {:.4f}"

        log_string(log, test_logs.format(i, mtest_loss, mtest_mape,
                                         mtest_rmse))
    log_string(
        log,
        "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log,
               "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


def initial_dataset():
    train_dataset = Air_dataset(data=args.dataset,
                                flag='train',
                                seq_len=args.history,
                                pred_len=args.horizon)
    print('train dataset: ', len(train_dataset))
    val_dataset = Air_dataset(data=args.dataset,
                              flag='val',
                              seq_len=args.history,
                              pred_len=args.horizon)
    print('val dataset: ', len(val_dataset))
    test_dataset = Air_dataset(data=args.dataset,
                               flag='test',
                               seq_len=args.history,
                               pred_len=args.horizon)
    print('test dataset: ', len(test_dataset))
    train_loader = data.DataLoader(train_dataset,
                                   args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   drop_last=True,
                                   pin_memory=True)
    valid_loader = data.DataLoader(val_dataset,
                                   args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   drop_last=True,
                                   pin_memory=True)
    test_loader = data.DataLoader(test_dataset,
                                  args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    return test_dataset, test_loader, train_dataset, train_loader, val_dataset, valid_loader


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    log_string(log, "total time: %.1fmin" % ((end - start) / 60))
    log.close()
