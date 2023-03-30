import os
import time
import argparse
from tqdm import tqdm
import pickle
import copy
import sys
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch.nn.functional as F

from genotypes import CellLinkDownPos, CellLinkUpPos, CellPos
from nas_model import get_models

sys.path.append('../')
from datasets import datasets_dict
from utils import calc_parameters_count, get_logger
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import SoftDiceLoss, DiceLoss, BCEDiceLoss
from utils import MultiClassEntropyDiceLoss


def main(args):
    ############    init config ################
    #################### init logger ###################################
    log_dir = './search_exp/' + '/{}'.format(args.model) + \
              '/{}'.format(args.dataset) + '/{}_{}'.format(time.strftime('%Y%m%d-%H%M%S'), args.note)

    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Search'.format(args.model))
    args.save_path = log_dir
    args.save_tbx_log = args.save_path + '/tbx_log'
    writer = SummaryWriter(args.save_tbx_log)
    ##################### init device #################################
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
    args.multi_gpu = args.gpus > 1 and torch.cuda.is_available()
    args.device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.enabled = True
        cudnn.benchmark = True
    setting = {k: v for k, v in args._get_kwargs()}
    logger.info(setting)

    ####################### init dataset ###########################################
    logger.info("Dataset for search is {}".format(args.dataset))
    train_dataset = datasets_dict[args.dataset](args, args.dataset_root, split='train')
    val_dataset = datasets_dict[args.dataset](args, args.dataset_root, split='valid')

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # init loss
    if args.loss == 'bce':
        criterion = nn.BCELoss()
    elif args.loss == 'bcelog':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "softdice":
        criterion = SoftDiceLoss()
    elif args.loss == 'bcedice':
        criterion = BCEDiceLoss()
    elif args.loss == 'multibcedice':
        criterion = MultiClassEntropyDiceLoss(dice_weight=args.dice_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        logger.info("load criterion to gpu !")
    criterion = criterion.to(args.device)

    ######################## init model ############################################
    switches_normal = []
    switches_down = []
    switches_up = []
    nums_mixop = sum([2 + i for i in range(args.meta_node_num)])
    for i in range(nums_mixop):
        switches_normal.append([True for j in range(len(CellPos))])
    for i in range(nums_mixop):
        switches_down.append([True for j in range(len(CellLinkDownPos))])
    for i in range(nums_mixop):
        switches_up.append([True for j in range(len(CellLinkUpPos))])

    # stage0 pruning  stage 1 pruning, stage 2 (training)
    original_train_batch = args.train_batch
    original_val_batch = args.val_batch

    #############################select model########################
    args.model = "UnetLayer7"
    args.layers = 7
    sp_train_batch = original_train_batch
    sp_val_batch = original_val_batch
    sp_epoch = args.epochs
    sp_lr = args.lr
    early_fix_arch = args.early_fix_arch
    gen_max_child_flag = args.gen_max_child_flag
    random_sample = args.random_sample

    train_queue = data.DataLoader(train_dataset,
                                  batch_size=sp_train_batch,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                  pin_memory=True,
                                  num_workers=args.num_workers
                                  )
    val_queue = data.DataLoader(train_dataset,
                                batch_size=sp_train_batch,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                                pin_memory=True,
                                num_workers=args.num_workers
                                )


    logger.info(
        "model:{} epoch:{} lr:{} train_batch:{} val_batch:{}".format(args.model, sp_epoch, sp_lr, sp_train_batch,
                                                                     sp_val_batch))

    model = get_models(args, switches_normal, switches_down, switches_up, early_fix_arch, gen_max_child_flag,
                       random_sample)

    for v in model.parameters():
        if v.requires_grad:
            if v.grad is None:
                v.grad = torch.zeros_like(v)
    model.alphas_up.grad = torch.zeros_like(model.alphas_up)
    model.alphas_down.grad = torch.zeros_like(model.alphas_down)
    model.alphas_normal.grad = torch.zeros_like(model.alphas_normal)
    model.alphas_network.grad = torch.zeros_like(model.alphas_network)

    wo_wd_params = []
    wo_wd_param_names = []
    network_params = []
    network_param_names = []

    for name, mod in model.named_modules():

        if isinstance(mod, nn.BatchNorm2d):
            for key, value in mod.named_parameters():
                wo_wd_param_names.append(name + '.' + key)


    for key, value in model.named_parameters():
        if "alphas" not in key:
            if value.requires_grad:
                if key in wo_wd_param_names:
                    wo_wd_params.append(value)  # 模块参数名字 权值
                else:
                    network_params.append(value)
                    network_param_names.append(key)  # 模块以外的参数

    weight_parameters = [
        {'params': network_params,
         'lr': args.lr,
         'weight_decay': args.weight_decay},
        {'params': wo_wd_params,
         'lr': args.lr,
         'weight_decay': 0.},
    ]

    save_model_path = os.path.join(args.save_path, 'singlepath')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if args.multi_gpu:
        logger.info('use: %d gpus', args.gpus)
        model = nn.DataParallel(model)
    model = model.to(args.device)
    logger.info('param size = %fMB', calc_parameters_count(model))
    # init optimizer for arch parameters and weight parameters
    # final stage, just train the network parameters
    optimizer_arch = torch.optim.Adam(model.arch_parameters(), lr=args.arch_lr, betas=(0.5, 0.999),
                                      weight_decay=args.arch_weight_decay)

    optimizer_weight = torch.optim.SGD(weight_parameters, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_weight, sp_epoch, eta_min=args.lr_min)

    #################################### train and val ########################
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint['epoch']
            optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
            optimizer_weight.load_state_dict(checkpoint['optimizer_weight'])
            model.load_alphas(checkpoint['alphas_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    max_value = 0
    for epoch in range(start_epoch, sp_epoch):

        scheduler.step()

        logger.info('################Epoch: %d lr %e######################', epoch, scheduler.get_last_lr()[0])

        if args.early_fix_arch:
            if len(model.fix_arch_normal_index.keys()) > 0:
                for key, value_lst in model.fix_arch_normal_index.items():
                    model.alphas_normal.data[key, :] = value_lst[1]

            sort_log_alpha_normal = torch.topk(F.softmax(model.alphas_normal.data, dim=-1), 2)  # 返回前两个alpha值
            argmax_index_normal = (sort_log_alpha_normal[0][:, 0] - sort_log_alpha_normal[0][:, 1] >= 0.3)
            # print("1 and 2",sort_log_alpha,argmax_index)
            for id in range(argmax_index_normal.size(0)):
                if argmax_index_normal[id] == 1 and id not in model.fix_arch_normal_index.keys():
                    model.fix_arch_normal_index[id] = [sort_log_alpha_normal[1][id, 0].item(),
                                                       model.alphas_normal.detach().clone()[id, :]]

            if len(model.fix_arch_down_index.keys()) > 0:
                for key, value_lst in model.fix_arch_down_index.items():
                    model.alphas_down.data[key, :] = value_lst[1]
            sort_log_alpha_down = torch.topk(F.softmax(model.alphas_down.data, dim=-1), 2)  # 返回前两个alpha值
            argmax_index_down = (sort_log_alpha_down[0][:, 0] - sort_log_alpha_down[0][:, 1] >= 0.3)
            # print("1 and 2",sort_log_alpha,argmax_index)
            for id in range(argmax_index_down.size(0)):
                if argmax_index_down[id] == 1 and id not in model.fix_arch_down_index.keys():
                    model.fix_arch_down_index[id] = [sort_log_alpha_down[1][id, 0].item(),
                                                     model.alphas_down.detach().clone()[id, :]]

            if len(model.fix_arch_up_index.keys()) > 0:
                for key, value_lst in model.fix_arch_up_index.items():
                    model.alphas_up.data[key, :] = value_lst[1]
            sort_log_alpha_up = torch.topk(F.softmax(model.alphas_up.data, dim=-1), 2)  # 返回前两个alpha值
            argmax_index_up = (sort_log_alpha_up[0][:, 0] - sort_log_alpha_up[0][:, 1] >= 0.3)
            # print("1 and 2",sort_log_alpha,argmax_index)
            for id in range(argmax_index_up.size(0)):
                if argmax_index_up[id] == 1 and id not in model.fix_arch_up_index.keys():
                    model.fix_arch_up_index[id] = [sort_log_alpha_up[1][id, 0].item(),
                                                   model.alphas_up.detach().clone()[id, :]]
        # train
        if epoch < args.arch_after:
            trainloss = train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch,
                              train_arch=False)
        else:
            trainloss = train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch,
                              train_arch=True)

        logger.info("Epoch:{} WeightLoss:{:.3f}  ArchLoss:{:.3f}".format(epoch, trainloss[0], trainloss[1]))
        # write
        writer.add_scalar('Train/W_loss', trainloss[0], epoch)
        writer.add_scalar('Train/cell_loss', trainloss[1], epoch)
        # writer.add_scalar('Train/n_loss', trainloss[2], epoch)

        # infer
        if (epoch + 1) % args.infer_epoch == 0:
            genotype = model.genotype()
            logger.info('genotype = %s', genotype)
            val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
            logger.info("ValLoss:{:.3f} ValAcc:{:.3f}  ValDice:{:.3f} ValJc:{:.3f}".format(val_loss, vmacc, vmd, vmjc))
            writer.add_scalar('Val/loss', val_loss, epoch)

            if args.gen_max_child:
                args.gen_max_child_flag = True
                val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
                logger.info(
                    "ValLoss2:{:.3f} ValAcc2:{:.3f}  ValDice2:{:.3f} ValJc2:{:.3f}".format(val_loss, vmacc, vmd, vmjc))
                writer.add_scalar('Val/loss', val_loss, epoch)
                args.gen_max_child_flag = False

            is_best = True if (vmjc >= max_value) else False
            max_value = max(max_value, vmjc)
            state = {
                'epoch': epoch,
                # 'optimizer_cellarch': optimizer_cellarch.state_dict(),
                'optimizer_arch': optimizer_arch.state_dict(),
                'optimizer_weight': optimizer_weight.state_dict(),
                'scheduler': scheduler.state_dict(),
                'state_dict': model.state_dict(),
                'alphas_dict': model.alphas_dict(),
            }

            logger.info("epoch:{} best:{} max_value:{}".format(epoch, is_best, max_value))
            if not is_best:
                torch.save(state, os.path.join(save_model_path, "checkpoint.pth.tar"))
            else:
                torch.save(state, os.path.join(save_model_path, "checkpoint.pth.tar"))
                torch.save(state, os.path.join(save_model_path, "model_best.pth.tar"))

        # one stage end, we should change the operations num (divided 2)
    weight_down = F.softmax(model.arch_parameters()[0], dim=-1).data.cpu().numpy()
    weight_up = F.softmax(model.arch_parameters()[1], dim=-1).data.cpu().numpy()
    weight_normal = F.softmax(model.arch_parameters()[2], dim=-1).data.cpu().numpy()
    weight_network = F.softmax(model.arch_parameters()[3], dim=-1).data.cpu().numpy()
    logger.info("alphas_down: \n{}".format(weight_down))
    logger.info("alphas_up: \n{}".format(weight_up))
    logger.info("alphas_normal: \n{}".format(weight_normal))
    logger.info("alphas_network: \n{}".format(weight_network))

    genotype = model.genotype()
    logger.info('Genotype: {}'.format(genotype))

    writer.close()


def train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch, train_arch):
    w_tloss_recoder = AverageMeter()
    a_loss_recoder = AverageMeter()

    model.train()
    for step, (input, target, _) in tqdm(enumerate(train_queue)):
        # n,c h w
        input = input.to(args.device)
        target = target.to(args.device)
        optimizer_weight.zero_grad()
        preds = model(input, target, criterion)
        assert isinstance(preds, list)
        torch.cuda.empty_cache()
        if args.deepsupervision:
            assert isinstance(preds, list) or isinstance(preds, tuple)
            tloss = 0
            for index in range(len(preds)):
                subtloss = criterion(preds[index].view(preds[index].size(0), -1), target.view(target.size(0), -1))
                tloss += subtloss
            tloss = tloss * preds[-1].size(0)
        else:
            tloss = criterion(preds[-1].view(preds[-1].size(0), -1), target.view(target.size(0), -1))
            tloss = tloss * preds[-1].size(0)

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
        optimizer_weight.step()
        w_tloss_recoder.update(tloss.item(), 1)

        # update cell arch parameters
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above.
            try:
                input_search, target_search, _ = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(val_queue)
                input_search, target_search, _ = next(valid_queue_iter)
            input_search = input_search.to(args.device)
            target_search = target_search.to(args.device)
            optimizer_arch.zero_grad()
            archs_preds = model(input_search, target_search, criterion)
            assert isinstance(archs_preds, list) or isinstance(archs_preds, tuple)
            torch.cuda.empty_cache()
            if args.deepsupervision:
                atloss = 0
                for index in range(len(archs_preds)):
                    asubtloss = criterion(archs_preds[index].view(archs_preds[index].size(0), -1),
                                          target_search.view(target_search.size(0), -1))
                    atloss += asubtloss
                atloss = atloss * archs_preds[-1].size(0)
            else:
                atloss = criterion(archs_preds[-1].view(archs_preds[-1].size(0), -1),
                                   target_search.view(target_search.size(0), -1))
                atloss = atloss * archs_preds[-1].size(0)

            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)
            optimizer_arch.step()
            a_loss_recoder.update(atloss.item(), 1)

    weight_loss_avg = w_tloss_recoder.avg
    if train_arch:
        a_loss_avg = a_loss_recoder.avg

    else:
        a_loss_avg = 0

    return weight_loss_avg, a_loss_avg


def infer(args, model, val_queue, criterion):
    OtherVal = BinaryIndicatorsMetric()
    tloss_r = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target, _) in enumerate(val_queue):
            input = input.to(args.device)
            target = target.to(args.device)
            preds = model(input, target, criterion)
            preds = [pred.view(pred.size(0), -1) for pred in preds]
            target = target.view(target.size(0), -1)
            assert isinstance(preds, list) or isinstance(preds, tuple)
            if args.deepsupervision:
                tloss = 0
                for index in range(len(preds)):
                    subtloss = criterion(preds[index].view(preds[index].size(0), -1), target.view(target.size(0), -1))
                    tloss += subtloss
                tloss = tloss * preds[-1].size(0)
            else:
                tloss = criterion(preds[-1].view(preds[-1].size(0), -1), target.view(target.size(0), -1))
                tloss = tloss * preds[-1].size(0)
            tloss_r.update(tloss.item(), 1)
            # n,1,h,w    n h w
            OtherVal.update(labels=target, preds=preds[-1].view(preds[-1].size(0), -1), n=1)

        return tloss_r.avg, OtherVal.get_avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet serieas Search')
    # Add default argument
    parser.add_argument('--model', type=str, default='Nas_Search_Unet',
                        help='Model to train and evaluation')
    parser.add_argument('--note', type=str, default='_', help="folder name note")
    parser.add_argument('--dataset', type=str, default='chaos',
                        help='Model to train and evaluation')
    parser.add_argument('--dataset_root', type=str,
                        default='/content/drive/MyDrive/Dataset/Mixsearch/CHAOS_CT',
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--epochs', type=int, default=60, help="search epochs")
    parser.add_argument('--train_batch', type=int, default=6, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=6, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader numworkers")
    parser.add_argument('--train_portion', type=float, default=0.5, help="dataloader numworkers")

    # search network setting
    parser.add_argument('--num_classes', type=int, default=1, help="output feature channel")
    parser.add_argument('--input_c', type=int, default=1, help="input img channel")
    parser.add_argument('--init_channel', type=int, default=16, help="init_channel for first leavel search cell")
    parser.add_argument('--meta_node_num', type=int, default=4, help="middle_nodes")
    parser.add_argument('--layers', type=int, default=4, help="layers")
    parser.add_argument('--use_sharing', type=bool, default=True,
                        help="The down op and up op have same normal operations")
    parser.add_argument('--depth', type=int, default=7, help="UnetFabrics`s layers and depth ")
    parser.add_argument('--double_down_channel', action='store_true', default=True, help="double_down_channel")
    parser.add_argument('--dropout_prob', type=float, default=0, help="dropout_prob")
    parser.add_argument('--use_softmax_head', type=bool, default=False, help='use_softmax_head')

    # model and device setting
    parser.add_argument('--init_weight_type', type=str, default="kaiming", help="the model init ")
    parser.add_argument('--arch_after', type=int, default=30,
                        help=" the first arch_after epochs without arch parameters traing")
    parser.add_argument('--infer_epoch', type=int, default=4, help=" val freq(epoch) ")
    parser.add_argument('--compute_freq', type=int, default=40, help=" compute freq(epoch) ")
    parser.add_argument('--gpus', type=int, default=1, help=" use cuda or not ")
    parser.add_argument('--grad_clip', type=int, default=0, help=" grid clip to ignore grad boom")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")

    # seatch setting
    parser.add_argument('--loss', type=str, choices=['bcedice', 'bce', 'bcelog', 'dice', 'softdice', 'multibcedice'],
                        default="bcedice", help="loss name ")
    parser.add_argument('--dice_weight', type=int, default=10, help="dice loss weight in total loss")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adm'], default='sgd',
                        help=" model_optimizer ! ")
    parser.add_argument('--deepsupervision', action='store_true', default=True, help=" deepsupervision nas uent ")
    # lr
    parser.add_argument('--lr', type=float, default=0.025, help="weight parameters lr ")
    parser.add_argument('--lr_min', type=float, default=1e-5, help=" min arch parameters lr  ")
    parser.add_argument('--weight_decay', type=float, default=3e-4, help=" for weight parameters lr  ")
    parser.add_argument('--arch_lr', type=float, default=1e-3, help="arch parameters lr ")
    parser.add_argument('--arch_weight_decay', type=float, default=0, help=" for arch parameters lr ")
    parser.add_argument('--momentum', type=float, default=0.9, help=" momentum  ")
    # resume
    parser.add_argument('--resume', type=str, default=None, help=" resume file path")

    # DSNAS
    parser.add_argument('--early_fix_arch', action='store_true', default=True, help='bn affine flag')
    parser.add_argument('--gen_max_child', action='store_true', default=True,
                        help='generate child network by argmax(alpha)')
    parser.add_argument('--gen_max_child_flag', action='store_true', default=False,
                        help='flag of generating child network by argmax(alpha)')
    parser.add_argument('--random_sample', action='store_true', default=False, help='true if sample randomly')
    args = parser.parse_args()
    main(args)

