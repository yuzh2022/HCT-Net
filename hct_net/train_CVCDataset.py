import os
import time
import argparse
from tqdm import tqdm
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch.nn.functional as F

from genotypes import CellLinkDownPos, CellLinkUpPos, CellPos
from nas_model import get_models

sys.path.append('../')
from datasets import get_dataloder, datasets_dict
from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import BCEDiceLoss, SoftDiceLoss, DiceLoss
from utils import LRScheduler


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

    original_train_batch = args.train_batch
    original_val_batch = args.val_batch


    #############################select model################################
    args.model = "UnetLayer7"
    args.layers = 7
    sp_train_batch = original_train_batch
    sp_val_batch = original_val_batch
    sp_lr = args.lr
    sp_epoch = args.epochs
    early_fix_arch = args.early_fix_arch
    gen_max_child_flag = args.gen_max_child_flag
    random_sample = args.random_sample

    ###################################dataset#########################################
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
    # print("1",model.named_modules())
    for name, mod in model.named_modules():
        # print("1:",name,mod)
        if isinstance(mod, nn.BatchNorm2d):
            for key, value in mod.named_parameters():
                wo_wd_param_names.append(name + '.' + key)
            # print("pa:",wo_wd_param_names)

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
         'weight_decay': args.weight_decay },
        {'params': wo_wd_params,
         'lr': args.lr,
         'weight_decay': 0.},
    ]


    save_model_path = os.path.join(args.save_path, 'single')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if args.multi_gpu:
        logger.info('use: %d gpus', args.gpus)
        model = nn.DataParallel(model)
    model = model.to(args.device)
    logger.info('param size = %fMB', calc_parameters_count(model))

    # init optimizer for arch parameters and weight parameters
    optimizer_arch = torch.optim.Adam(model.arch_parameters(), lr=args.arch_lr, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    optimizer_weight = torch.optim.SGD(weight_parameters, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_weight, sp_epoch, eta_min=args.lr_min)

    #################################### train and val ########################
    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint['epoch']
            #optimizer_cellarch.load_state_dict(checkpoint['optimizer_cellarch'])
            optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
            optimizer_weight.load_state_dict(checkpoint['optimizer_weight'])
            model.load_alphas(checkpoint['alphas_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))




    max_value=0
    for epoch in range(start_epoch, sp_epoch):

        scheduler.step()
        logger.info('################Epoch: %d lr %e######################', epoch, scheduler.get_lr()[0])

        if args.early_fix_arch:
            if len(model.fix_arch_normal_index.keys()) > 0:
                for key, value_lst in model.fix_arch_normal_index.items():
                    model.alphas_normal.data[key, :] = value_lst[1]

            sort_log_alpha_normal = torch.topk(F.softmax(model.alphas_normal.data, dim=-1), 2)  # 返回前两个alpha值
            argmax_index_normal = (sort_log_alpha_normal[0][:, 0] - sort_log_alpha_normal[0][:, 1] >= 0.3)

            for id in range(argmax_index_normal.size(0)):
                if argmax_index_normal[id] == 1 and id not in model.fix_arch_normal_index.keys():
                    model.fix_arch_normal_index[id] = [sort_log_alpha_normal[1][id, 0].item(),
                                                       model.alphas_normal.detach().clone()[id, :]]

            if len(model.fix_arch_down_index.keys()) > 0:
                for key, value_lst in model.fix_arch_down_index.items():
                    model.alphas_down.data[key, :] = value_lst[1]
            sort_log_alpha_down = torch.topk(F.softmax(model.alphas_down.data, dim=-1), 2)  # 返回前两个alpha值
            argmax_index_down = (sort_log_alpha_down[0][:, 0] - sort_log_alpha_down[0][:, 1] >= 0.3)

            for id in range(argmax_index_down.size(0)):
                if argmax_index_down[id] == 1 and id not in model.fix_arch_down_index.keys():
                    model.fix_arch_down_index[id] = [sort_log_alpha_down[1][id, 0].item(),
                                                     model.alphas_down.detach().clone()[id, :]]

            if len(model.fix_arch_up_index.keys()) > 0:
                for key, value_lst in model.fix_arch_up_index.items():
                    model.alphas_up.data[key, :] = value_lst[1]
            sort_log_alpha_up = torch.topk(F.softmax(model.alphas_up.data, dim=-1), 2)  # 返回前两个alpha值
            argmax_index_up = (sort_log_alpha_up[0][:, 0] - sort_log_alpha_up[0][:, 1] >= 0.3)

            for id in range(argmax_index_up.size(0)):
                if argmax_index_up[id] == 1 and id not in model.fix_arch_up_index.keys():
                    model.fix_arch_up_index[id] = [sort_log_alpha_up[1][id, 0].item(),
                                                   model.alphas_up.detach().clone()[id, :]]
        if epoch < args.arch_after:
          weight_loss_avg, arch_loss_avg, mr, ms, mp, mf, mjc, md, macc,epoch_time = train(args, train_queue, val_queue,model, criterion,optimizer_weight, optimizer_arch,
                                                                              train_arch=False)
        else:
          weight_loss_avg, arch_loss_avg, mr, ms, mp, mf, mjc, md, macc,epoch_time = train(args, train_queue, val_queue,model, criterion,optimizer_weight, optimizer_arch,
                                                                              train_arch=True)

        logger.info("Epoch:{} WeightLoss:{:.3f}  ArchLoss:{:.3f}".format(epoch, weight_loss_avg, arch_loss_avg))
        logger.info("         Acc:{:.3f}   Dice:{:.3f}  Jc:{:.3f}".format(macc, md, mjc))

        # write
        writer.add_scalar('Train/W_loss', weight_loss_avg, epoch)
        writer.add_scalar('Train/A_loss', arch_loss_avg, epoch)
        writer.add_scalar('Train/Dice', md, epoch)
#        writer.add_scalar('Train/time_each_epoch',epoch_time , epoch)

        # infer
        if (epoch + 1) % args.infer_epoch == 0:
            genotype = model.genotype()
            logger.info('genotype = %s', genotype)
            val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
            logger.info("ValLoss1:{:.3f} ValAcc1:{:.3f}  ValDice1:{:.3f} ValJc1:{:.3f}".format(val_loss, vmacc, vmd, vmjc))
            #writer.add_scalar('Val/loss1', val_loss, epoch)
            writer.add_scalar('Val/coffecient', vmjc, epoch)
            writer.add_scalar('Val/JC1', vmd, epoch)

            if args.gen_max_child:
                args.gen_max_child_flag = True
                val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
                logger.info("ValLoss2:{:.3f} ValAcc2:{:.3f}  ValDice2:{:.3f} ValJc2:{:.3f}".format(val_loss, vmacc, vmd, vmjc))
                writer.add_scalar('Val/loss', val_loss, epoch)
                args.gen_max_child_flag = False

            is_best = True if (vmjc >= max_value) else False
            max_value = max(max_value, vmjc)
            state = {
                'epoch': epoch,
                'max_value': max_value,
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
    weight_down = F.softmax(model.arch_parameters()[0], dim=-1).data.cpu().numpy()  #
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
    Train_recoder = BinaryIndicatorsMetric()
    w_loss_recoder = AverageMeter()
    a_loss_recoder = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()

    for step, (input, target, _) in enumerate(train_queue):
        data_time.update(time.time() - end)

        input = input.to(args.device)
        target = target.to(args.device)

        # input is B C H W   target is B,1,H,W  preds: B,1,H,W
        optimizer_weight.zero_grad()
        preds = model(input,target,criterion)
        assert isinstance(preds, list)
        preds = [pred.view(pred.size(0), -1) for pred in preds]

        target = target.view(target.size(0), -1)

        torch.cuda.empty_cache()

        Train_recoder.update(labels=target, preds=preds[-1], n=1)

        if args.deepsupervision:
            for i in range(len(preds)):
                if i == 0:
                    target1_loss = criterion(preds[i], target)
                target1_loss += criterion(preds[i], target)
        else:
            target1_loss = criterion(preds[-1], target)

        w_loss = target1_loss

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)

        optimizer_weight.step()

        w_loss_recoder.update(w_loss.item(), 1)

        # get all the indicators
        Train_recoder.update(labels=target, preds=preds[-1], n=1)

        # update network arch parameters
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
            archs_preds = model(input_search,target_search,criterion)
            archs_preds =[pred.view(pred.size(0), -1) for pred in archs_preds]
            target_search = target_search.view(target_search.size(0), -1)
            torch.cuda.empty_cache()
            if args.deepsupervision:
                for i in range(len(archs_preds)):
                    if i == 0:
                        a_loss = criterion(archs_preds[i], target_search)
                    a_loss += criterion(archs_preds[i], target_search)
            else:
                a_loss = criterion(archs_preds[-1], target_search)

            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)


            optimizer_arch.step()

            a_loss_recoder.update(a_loss.item(), 1)

    weight_loss_avg = w_loss_recoder.avg
    if train_arch:
        arch_loss_avg = a_loss_recoder.avg
    else:
        arch_loss_avg = 0
    mr, ms, mp, mf, mjc, md, macc = Train_recoder.get_avg
    batch_time.update(time.time() - end)
    end = time.time()
    print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time))

    return weight_loss_avg, arch_loss_avg, mr, ms, mp, mf, mjc, md, macc,batch_time


def infer(args, model, val_queue, criterion):
    batch_time = AverageMeter()
    OtherVal = BinaryIndicatorsMetric()
    val_loss = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (input, target, _) in tqdm(enumerate(val_queue)):
            input = input.to(args.device)
            target = target.to(args.device)
            preds = model(input,target,criterion)
            preds = [pred.view(pred.size(0), -1) for pred in preds]
            target = target.view(target.size(0), -1)
            if args.deepsupervision:
                for i in range(len(preds)):
                    if i == 0:
                        loss = criterion(preds[i], target)
                    loss += criterion(preds[i], target)
            else:
                loss = criterion(preds[-1], target)
            val_loss.update(loss.item(), 1)
            OtherVal.update(labels=target, preds=preds[-1], n=1)

            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time))
        return val_loss.avg, OtherVal.get_avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet serieas Search')
    # Add default argument
    parser.add_argument('--model', type=str, default='Nas_Search_Unet',
                        help='Model to train and evaluation')
    parser.add_argument('--note', type=str, default='_', help="folder name note")
    parser.add_argument('--dataset', type=str, default='isic2018',
                        help='Model to train and evaluation')
    parser.add_argument('--dataset_root', type=str,
                        default='/content/drive/MyDrive/Dataset/Mixsearch/ISIC2018/Task1_apply',
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--epochs', type=int, default=60, help="search epochs")
    parser.add_argument('--train_batch', type=int, default=4, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=4, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=4, help="dataloader numworkers")
    parser.add_argument('--train_portion', type=float, default=0.5, help="dataloader numworkers")

    # search network setting
    parser.add_argument('--num_classes', type=int, default=1, help="output feature channel")
    parser.add_argument('--input_c', type=int, default=3, help="input img channel")
    parser.add_argument('--init_channel', type=int, default=16, help="init_channel for first leavel search cell")
    parser.add_argument('--meta_node_num', type=int, default=4, help="middle_nodes")
    parser.add_argument('--layers', type=int, default=9, help="layers")
    parser.add_argument('--use_sharing', type=bool, default=True,
                        help="The down op and up op have same normal operations")
    parser.add_argument('--depth', type=int, default=4, help="UnetFabrics`s layers and depth ")
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
    parser.add_argument('--grad_clip', type=int, default=1, help=" grid clip to ignore grad boom")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")

    # seatch setting
    parser.add_argument('--loss', type=str, choices=['bcedice', 'bce', 'bcelog', 'dice', 'softdice', 'multibcedice'],
                        default="bcelog", help="loss name ")
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
    parser.add_argument('--resume', type=str, default='/content/drive/MyDrive/MIxSearch12.31_dynamic_transformer/nas_search/search_exp/Nas_Search_Unet/isic2018/20220413-093911__/DSNAS/checkpoint.pth.tar', help=" resume file path")
    #DSNAS
    parser.add_argument('--early_fix_arch', action='store_true', default=True, help='bn affine flag')
    parser.add_argument('--gen_max_child', action='store_true', default=True,help='generate child network by argmax(alpha)')
    parser.add_argument('--gen_max_child_flag', action='store_true', default=False,help='flag of generating child network by argmax(alpha)')
    parser.add_argument('--random_sample', action='store_true', default=False, help='true if sample randomly')
    args = parser.parse_args()
    main(args)
