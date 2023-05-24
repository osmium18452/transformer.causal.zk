import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm

import argparse, pickle, platform, os, random

from DataPreprocessor import DataPreprocessor, TSDataset
from Transformer import Transformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alt_learning_rate', type=float, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-C', '--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('-c', '--cuda_device', type=int, default=0)
    parser.add_argument('-d', '--dataset', type=str, default='wht')
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-F', '--fixed_seed', action='store_true')
    parser.add_argument('-f', '--fudan', action='store_true')
    parser.add_argument('-m', '--gnn_map', type=str, default='T', help='L: laplacian, I, T')
    parser.add_argument('-G', '--gpu', action='store_true')
    parser.add_argument('-H', '--hidden_size', type=int, default=40)
    parser.add_argument('-i', '--input_window', type=int, default=60)
    parser.add_argument('-l', '--lr', type=float, default=0.001)
    parser.add_argument('-M', '--multiGPU', action='store_true')
    parser.add_argument('-o', '--model', type=str, default='cgt', help='cgt, transformer')
    parser.add_argument('-n', '--nhead', type=int, default=8)
    parser.add_argument('-N', '--normalize', type=str, default=None, help='std, minmax, zeromean')
    parser.add_argument('-P', '--positional_encoding', type=str, default='sinusoidal', help='zero,sin,sinusoidal')
    parser.add_argument('-p', '--predict_window', type=int, default=30)
    parser.add_argument('-s', '--save_path', type=str, default='save')
    parser.add_argument('-L', '--transformer_layers', type=int, default=8)
    parser.add_argument('-V', '--dont_validate', action='store_true')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('-t','--train_ratio', type=float, default=0.6)
    parser.add_argument('-v','--validate_ratio', type=float, default=0.2)
    args = parser.parse_args()

    use_cuda = args.gpu
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1
    if args.multiGPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda:' + str(args.cuda_device) if use_cuda else 'cpu')

    batch_size = args.batch_size
    input_time_window = args.input_window
    pred_time_window = args.predict_window
    nhead = args.nhead
    # hidden_size = args.hidden_size * nhead
    dataset = args.dataset
    total_epoch = args.epoch
    lr = args.lr
    transformer_layers = args.transformer_layers
    normalize = args.normalize
    train_ratio = args.train_ratio
    validate_ratio = args.validate_ratio
    cuda_device = args.cuda_device
    test_ratio = 1 - train_ratio - validate_ratio
    alt_learning_rate = args.alt_learning_rate
    step_size = args.step_size
    gnn_map = args.gnn_map
    stride = args.stride
    multiGPU = args.multiGPU
    fixed_seed = args.fixed_seed
    positional_encoding = args.positional_encoding
    save_path = args.save_path
    fudan = args.fudan
    validate = not args.dont_validate
    which_model = args.model

    if (multiGPU and local_rank == 0) or not multiGPU:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(args)

    # fix random seed
    if multiGPU or fixed_seed:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if multiGPU:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")

    if platform.system() == 'Windows':
        # data_dir = 'C:\\Users\\17110\\Desktop\\ts forecasting\\dataset\\pkl'
        # map_dir = 'C:\\Users\\17110\\Desktop\\ts forecasting\\dataset\\map'
        data_dir = 'E:\\forecastdataset\\pkl'
        map_dir = 'E:\\forecastdataset\\map'
    else:
        if fudan:
            data_dir = '/remote-home/liuwenbo/pycproj/forecastdata//pkl/'
            map_dir = '/remote-home/liuwenbo/pycproj/forecastdata//map/'
        else:
            data_dir = '/home/icpc/pycharmproj/forecast.dataset/pkl/'
            map_dir = '/home/icpc/pycharmproj/forecast.dataset/map/'
    if dataset == 'wht':
        dataset = pickle.load(open(os.path.join(data_dir, 'wht.pkl'), 'rb'))
        causal_map = pickle.load(open(os.path.join(map_dir, 'wht.map.pkl'), 'rb'))
    else:
        dataset = None
        causal_map = None
        print('dataset not found')
        exit()
    # if multiGPU and local_rank !=0:
    #     dist.barrier()
    data_preprocessor = DataPreprocessor(dataset=dataset, input_time_window=input_time_window,
                                         output_time_window=pred_time_window,
                                         normalize=normalize, train_ratio=train_ratio, valid_ratio=validate_ratio,
                                         stride=stride, positional_encoding=positional_encoding)
    causal_map = torch.Tensor(causal_map)
    parent_list = []
    n_sensors = causal_map.shape[0]
    # print('n_sensors', n_sensors)
    for i in range(n_sensors):
        parent_list.append([i])
        for j in range(n_sensors):
            if causal_map[i][j] == 1:
                parent_list[i].append(j)
        # if (multiGPU and local_rank == 0) or not multiGPU:
        #     print(len(parent_list[-1]))
    train_input, train_tgt, train_gt = data_preprocessor.load_train_data()
    test_input, test_tgt, test_gt = data_preprocessor.load_test_data()
    valid_input, valid_tgt, valid_gt = data_preprocessor.load_validate_data()

    # train_dataloader_list = []
    # test_dataloader_list = []
    # validate_dataloader_list = []
    # transformer_list=[]
    # print(train_tgt.shape)
    result = []
    for i in range(n_sensors):
    # for i in [11]:
        train_set = TSDataset(train_input, train_tgt, train_gt, parent_list[i])
        test_set = TSDataset(test_input, test_tgt, test_gt, parent_list[i])
        valid_set = TSDataset(valid_input, valid_tgt, valid_gt, parent_list[i])
        train_loader = DataLoader(train_set, sampler=DistributedSampler(train_set) if multiGPU else None,
                                  batch_size=batch_size, shuffle=False if multiGPU else True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        validate_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        # train_dataloader_list.append(train_loader)
        # test_dataloader_list.append(test_loader)
        # validate_dataloader_list.append(validate_loader)
        # transformer_list.append(Transformer(len(parent_list[i])))
        model=Transformer(len(parent_list[i]),n_heads=nhead)
        if use_cuda:
            model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if alt_learning_rate is not None:
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=alt_learning_rate)
        loss_fn = torch.nn.MSELoss()
        pbar_iter=None
        pbar_epoch=None
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_epoch = tqdm(total=total_epoch, ascii=True, dynamic_ncols=True,leave=False)
            pbar_epoch.set_description('sensor %d/%d' % (i+1,n_sensors))
        for epoch in range(total_epoch):
            # train
            model.train()
            total_iters = len(train_loader)
            if (multiGPU and local_rank == 0) or not multiGPU:
                pbar_iter = tqdm(total=total_iters, ascii=True, leave=False, dynamic_ncols=True)
                pbar_iter.set_description('training')
            for input_batch, tgt, gt in train_loader:
                if use_cuda:
                    input_batch = input_batch.to(device)
                    tgt = tgt.to(device)
                    gt = gt.to(device)
                optimizer.zero_grad()
                # print(input_batch.shape,tgt.shape)
                output = model(input_batch, tgt)
                # print('output, gt shape',output.shape,gt.shape)
                loss = loss_fn(output, gt)
                loss.backward()
                optimizer.step()
                if (multiGPU and local_rank == 0) or not multiGPU:
                    pbar_iter.set_postfix_str('loss: %.4f' % (loss.item()))
                    pbar_iter.update(1)
            if (multiGPU and local_rank == 0) or not multiGPU:
                pbar_iter.close()
            # dist.barrier()

            # validate
            model.eval()
            if ((multiGPU and local_rank == 0) or not multiGPU) and validate:
                with torch.no_grad():
                    total_iters = len(validate_loader)
                    pbar_iter = tqdm(total=total_iters, ascii=True, leave=False, dynamic_ncols=True)
                    pbar_iter.set_description('validating')
                    output_list = []
                    gt_list = []
                    for input_batch, tgt, gt in validate_loader:
                        if use_cuda:
                            input_batch = input_batch.to(device)
                            tgt = tgt.to(device)
                        output_list.append(model(input_batch,  tgt).cpu())
                        gt_list.append(gt)
                        pbar_iter.update(1)
                    pbar_iter.close()
                    output = torch.cat(output_list, dim=0)
                    ground_truth = torch.cat(gt_list, dim=0)
                    loss = loss_fn(output, ground_truth)
                    pbar_epoch.set_postfix_str('valid loss: %.4f' % (loss.item()))
                    # print('pbar epoch upd')
                    pbar_epoch.update()
            if multiGPU:
                dist.barrier()
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_epoch.close()

        # test
        if (multiGPU and local_rank == 0) or not multiGPU:
            total_iters = len(test_loader)
            pbar_iter = tqdm(total=total_iters, ascii=True, dynamic_ncols=True,leave=False)
            pbar_iter.set_description('testing, sensor%d/%d'%(i+1,n_sensors))
            output_list = []
            gt_list=[]
            with torch.no_grad():
                model.eval()
                for input_batch, tgt, gt in test_loader:
                    if use_cuda:
                        input_batch = input_batch.to(device)
                        tgt = tgt.to(device)
                    output_list.append(model(input_batch, tgt).cpu())
                    gt_list.append(gt)
                    pbar_iter.update(1)
            pbar_iter.close()
            output = torch.cat(output_list, dim=0)
            ground_truth=torch.cat(gt_list,dim=0)
            loss = loss_fn(output, ground_truth)
            result.append(loss.item())
            print('\033[31mloss of sensor %d/%d:\033[0m'%(i+1,n_sensors),loss.item())
    print('\033[32m',np.mean(result),'\033[0m')
