import os
import time
import argparse
from tqdm import tqdm
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import sys
from nas_hybrid_CNN_Transformer_prune import BuildSinglePthPrune_l7

sys.path.append('../')
from datasets import get_dataloder, datasets_dict
from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info, get_model_complexity_info
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import BCEDiceLoss, SoftDiceLoss, DiceLoss


def main(args):
    #################### init logger ###################################
    log_dir = './logs/' + '{}'.format(args.dataset) + '/{}_{}_{}'.format(args.model, time.strftime('%Y%m%d-%H%M%S'),
                                                                         args.note)
    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Train'.format(args.model))
    # setting
    args.save_path = log_dir
    args.save_tbx_log = args.save_path + '/tbx_log'
    writer = SummaryWriter(args.save_tbx_log)
    ##################### init device #################################
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.benchmark = True
    ####################### init dataset ###########################################
    train_loader = get_dataloder(args, split_flag="train")
    val_loader = get_dataloder(args, split_flag="valid")
    ######################## init model ############################################

    # model
    # get the network parameters
    if args.model == "layer7_double_deep":
        args.deepsupervision = True
        args.double_down_channel = True
        args.genotype_name = 'layer7_double_deep'
        model_alphas = None
        genotype = eval('genotypes.%s' % args.genotype_name)
        model = BuildSinglePthPrune_l7(
            genotype=genotype,
            input_c=args.in_channels,
            c=args.init_channels,
            num_classes=args.nclass,
            meta_node_num=args.middle_nodes,
            layers=args.layers,
            dp=args.dropout_prob,
            use_sharing=args.use_sharing,
            double_down_channel=args.double_down_channel,
            aux=args.aux
        )


    if torch.cuda.device_count() > 1 and args.use_cuda:
        logger.info('use: %d gpus', torch.cuda.device_count())
        model = nn.DataParallel(model)
    setting = {k: v for k, v in args._get_kwargs()}
    logger.info(setting)
    logger.info(genotype)
    logger.info('param size = %fMB', calc_parameters_count(model))

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
    else:
        criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        logger.info("load model and criterion to gpu !")
        model = model.to(args.device)
        criterion = criterion.to(args.device)
    # init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    # init schedulers  Steplr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)


    ############################### check resume #########################
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    #################################### train and val ########################
    max_value = 0
    for epoch in range(start_epoch, args.epoch):
        # lr=adjust_learning_rate(args,optimizer,epoch)
        scheduler.step()
        # train
        if args.deepsupervision:
            mean_loss, value1, value2 = train(args, model_alphas, model, criterion, train_loader, optimizer)
            mr, ms, mp, mf, mjc, md, macc = value1
            logger.info(
                "Epoch:{} Train_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, mean_loss, macc, md, mjc))
            writer.add_scalar('Train/dDice', md, epoch)
            writer.add_scalar('Train/dJc', mjc, epoch)
        else:
            mean_loss, value1 = train(args, model_alphas, model, criterion, train_loader,
                                      optimizer)
            mr, ms, mp, mf, mjc, md, macc = value1
            logger.info(
                "Epoch:{} Train_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, mean_loss, macc, md, mjc))
        # write
        writer.add_scalar('Train/Loss', mean_loss, epoch)

        # val
        if args.deepsupervision:
            vmean_loss, valuev1, valuev2 = infer(args, model_alphas, model, criterion, val_loader)
            vmr, vms, vmp, vmf, vmjc, vmd, vmacc = valuev1
            logger.info(
                "Epoch:{} Val_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, vmean_loss, vmacc, vmd, vmjc))

        else:
            vmean_loss, valuev1 = infer(args, model_alphas, model, criterion, val_loader)
            vmr, vms, vmp, vmf, vmjc, vmd, vmacc = valuev1
            logger.info(
                "Epoch:{} Val_Loss:{:.3f} Acc:{:.3f} Dice:{:.3f} Jc:{:.3f}".format(epoch, vmean_loss, vmacc, vmd, vmjc))

        is_best = True if vmjc >= max_value else False
        max_value = max(max_value, vmjc)
        writer.add_scalar('Val/Loss', vmean_loss, epoch)

        state = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'scheduler': model.state_dict(),
        }
        logger.info("epoch:{} best:{} max_value:{}".format(epoch, is_best, max_value))
        if not is_best:
            torch.save(state, os.path.join(args.save_path, "checkpoint.pth.tar"))
        else:
            torch.save(state, os.path.join(args.save_path, "checkpoint.pth.tar"))
            torch.save(state, os.path.join(args.save_path, "model_best.pth.tar"))

    writer.close()


def train(args, model_alphas, model, criterion, train_loader, optimizer):
    Train_recoder = BinaryIndicatorsMetric()
    Deep_Train_recoder = BinaryIndicatorsMetric()
    loss_recoder = AverageMeter()
    model.train()
    for step, (input, target, _) in tqdm(enumerate(train_loader)):
        input = input.to(args.device)
        target = target.to(args.device)
        # input is B C H W   target is B,1,H,W  preds: B,1,H,W
        optimizer.zero_grad()
        # [output1,...]
        if model_alphas is not None:
            preds_list = model(model_alphas, input)
        else:
            preds_list = model(input)
        preds_list = [pred.view(pred.size(0), -1) for pred in preds_list]
        target = target.view(target.size(0), -1)
        if args.deepsupervision:
            for i in range(len(preds_list)):
                if i == 0:
                    w_loss = criterion(preds_list[i], target)
                w_loss += criterion(preds_list[i], target)
        else:
            w_loss = criterion(preds_list[-1], target)
        w_loss.backward()

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        loss_recoder.update(w_loss.item(), 1)
        # get all the indicators
        # Deep_Train_recoder
        if args.deepsupervision:
            avg_pred = 0
            for pred in preds_list:
                avg_pred += pred
            avg_pred /= len(preds_list)
            Deep_Train_recoder.update(labels=target, preds=avg_pred, n=1)
        Train_recoder.update(labels=target, preds=preds_list[-1], n=1)
    if args.deepsupervision:
        return loss_recoder.avg, Train_recoder.get_avg, Deep_Train_recoder.get_avg
    else:
        return loss_recoder.avg, Train_recoder.get_avg


def infer(args, model_alphas, model, criterion, val_loader):
    OtherVal = BinaryIndicatorsMetric()
    DeepOtherVal = BinaryIndicatorsMetric()
    val_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target, _) in tqdm(enumerate(val_loader)):
            input = input.to(args.device)
            target = target.to(args.device)
            if model_alphas is not None:
                preds_list = model(model_alphas, input)
            else:
                preds_list = model(input)
            preds_list = [pred.view(pred.size(0), -1) for pred in preds_list]
            target = target.view(target.size(0), -1)
            if args.deepsupervision:
                for i in range(len(preds_list)):
                    if i == 0:
                        v_loss = criterion(preds_list[i], target)
                    v_loss += criterion(preds_list[i], target)
            else:
                v_loss = criterion(preds_list[-1], target)
            val_loss.update(v_loss.item(), 1)
            # get the deepsupervision
            if args.deepsupervision:
                avg_pred = 0
                for pred in preds_list:
                    avg_pred += pred
                avg_pred /= len(preds_list)
                DeepOtherVal.update(labels=target, preds=avg_pred, n=1)
            OtherVal.update(labels=target, preds=preds_list[-1], n=1)
        if args.deepsupervision:
            return val_loss.avg, OtherVal.get_avg, DeepOtherVal.get_avg
        else:
            return val_loss.avg, OtherVal.get_avg


if __name__ == '__main__':
    datasets_name = datasets_dict.keys()
    parser = argparse.ArgumentParser(description='Unet Nas baseline')
    # Add default argument
    parser.add_argument('--model', type=str, default='ISIC_1',
                        help='Model to train and evaluation')
    parser.add_argument('--alphas_model', type=str, default='', help="the genotype we get ")
    parser.add_argument('--dataset', type=str, default='isic2018', choices=datasets_name,
                        help='Model to train and evaluation')

    parser.add_argument('--note', type=str, default='_', help='train note')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--in_channels', type=int, default=3, help="input image channel ")
    parser.add_argument('--init_channels', type=int, default=16, help="cell init change channel ")
    parser.add_argument('--nclass', type=int, default=1, help="output feature channel")
    parser.add_argument('--epoch', type=int, default=300, help="epochs")
    parser.add_argument('--train_batch', type=int, default=8, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=8, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader numworkers")
    parser.add_argument('--layers', type=int, default=7, help='the layer of the nas search unet')
    parser.add_argument('--middle_nodes', type=int, default=4, help="middle_nodes ")
    parser.add_argument('--dropout_prob', type=int, default=0.0, help="dropout_prob")
    parser.add_argument('--print_freq', type=int, default=1, help=" print freq (iteras) ")
    parser.add_argument('--gpus', type=int, default=1, help=" use cuda or not ")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")
    parser.add_argument('--use_sharing', action='store_false', help='normal weight sharing')
    parser.add_argument('--double_down_channel', type=bool, default=True, help=" double_down_channel")
    parser.add_argument('--deepsupervision', type=bool, default=True, help=" like unet++")
    parser.add_argument('--grad_clip', type=int, default=0.1, help=" grid clip to ignore grad boom")

    # model special
    parser.add_argument('--aux', action='store_true', help=" deepsupervision of aux layer for  une")
    parser.add_argument('--aux_weight', type=float, default=1, help=" bce+aux_weight*aux_layer_loss")
    parser.add_argument('--genotype_name', type=str, default="ISIC_1", help="cell genotype")
    parser.add_argument('--loss', type=str, choices=['bce', 'bcelog', 'dice', 'softdice', 'becdice'],
                        default="becdice", help="loss name ")
    parser.add_argument('--model_optimizer', type=str, choices=['sgd', 'adm'], default='sgd',
                        help=" model_optimizer ! ")

    parser.add_argument('--lr', type=float, default=1e-3, help=" learning rate we use  ")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help=" learning rate we use  ")
    parser.add_argument('--momentum', type=float, default=0.9, help=" learning rate we use  ")
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 100],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.5, 0.5],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    # for adam
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    # resume
    parser.add_argument('--resume', type=str, default=None, help=" resume file path")
    args = parser.parse_args()
    main(args)


