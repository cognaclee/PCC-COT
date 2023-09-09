import torch
import os
import sys
sys.path.append(os.getcwd())
from models.autoencoder import AutoEncoder
from models.discriminator import Discriminator
from models.loss import GradientPenaltyLoss
import time
import argparse
from datetime import datetime
import torch.optim as optim
import torch.autograd as autograd
from models.utils import  AverageMeter, str2bool
from dataset.dataset import CompressDataset
from args.shapenet_args import parse_shapenet_args
from args.semantickitti_args import parse_semantickitti_args
from torch.optim.lr_scheduler import StepLR
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(model, quantize_latent_xyzs, discr, model_path):
    # load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict_g'])
    # update entropy bottleneck
    model.feats_eblock.update(force=True)
    if quantize_latent_xyzs == True:
        model.xyzs_eblock.update(force=True)
        
    discr.load_state_dict(checkpoint['state_dict_d'])
    start_epoch = checkpoint['epoch']+1
    best_chamfer_loss = checkpoint['best_chamfer_loss']

    return model, discr, start_epoch, best_chamfer_loss

def train(args):
    start = time.time()

    if args.batch_size > 1:
        print('The performance will degrade if batch_size is larger than 1!')

    if args.compress_normal == True:
        args.in_fdim = 6

    # load data
    train_dataset = CompressDataset(data_path=args.train_data_path, cube_size=args.train_cube_size, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)

    val_dataset = CompressDataset(data_path=args.val_data_path, cube_size=args.val_cube_size, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size)


    # create the model
    model = AutoEncoder(args)
    model = model.cuda()
    print('Training Arguments:', args)
    

    discr = Discriminator(args)
    discr = discr.cuda()
    print('Discriminator Architecture:', discr)
    
    if args.model_path is not None:
        model, discr, start_epoch, best_val_chamfer_loss = load_model(model, args.quantize_latent_xyzs, discr, args.model_path)
        print('The AutoEncoder model have resumed through the model in : ', args.model_path)
        idx = args.model_path.rfind('/')
        checkpoint_dir = args.model_path[:idx]
    else:
        # set up folders for checkpoints
        str_time = datetime.now().isoformat()
        print('Experiment Time:', str_time)
        checkpoint_dir = os.path.join(args.output_path, str_time, 'ckpt')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        best_val_chamfer_loss = float('inf')
        start_epoch = 0

    # optimizer for autoencoder
    parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=args.lr)
    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    # optimizer for entropy bottleneck
    aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer = optim.Adam(aux_parameters, lr=args.aux_lr)

    # optimizer for discriminator
    dis_optimizer = optim.Adam(discr.parameters(), lr=args.dis_lr)
    gradientPenalty =GradientPenaltyLoss(device='cuda')

    # train
    for epoch in range(start_epoch,args.epochs):
        epoch_loss = AverageMeter()
        epoch_chamfer_loss = AverageMeter()
        epoch_latent_xyzs_loss = AverageMeter()
        epoch_normal_loss = AverageMeter()
        epoch_bpp_loss = AverageMeter()
        epoch_aux_loss = AverageMeter()
        epoch_dis_g_loss = AverageMeter()
        epoch_dis_d_loss = AverageMeter()
        epoch_dis_p_loss = AverageMeter()
        
        model.train()
        discr.train()

        for i, input_dict in enumerate(train_loader):
            # input: (b, n, c)
            
            input = input_dict['xyzs'].cuda()
            # input: (b, c, n)
            input = input.permute(0, 2, 1).contiguous()

            # compress normal
            if args.compress_normal == True:
                normals = input_dict['normals'].cuda().permute(0, 2, 1).contiguous()
                input = torch.cat((input, normals), dim=1)

            # model forward
            decompressed_xyzs, loss, loss_items, bpp = model(input)
            
            epoch_loss.update(loss.item())
            epoch_chamfer_loss.update(loss_items['chamfer_loss'])
            epoch_latent_xyzs_loss.update(loss_items['latent_xyzs_loss'])
            epoch_normal_loss.update(loss_items['normal_loss'])
            epoch_bpp_loss.update(loss_items['bpp_loss'])
            

            fake = discr(decompressed_xyzs)
            real = discr(input).detach()
            dis_g_loss = args.gan_g_coe*torch.pow(fake-real,2)
            epoch_dis_g_loss.update(dis_g_loss.item())
            loss = loss + dis_g_loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the parameters of entropy bottleneck
            aux_loss = model.feats_eblock.loss()
            if args.quantize_latent_xyzs == True:
                aux_loss += model.xyzs_eblock.loss()
            epoch_aux_loss.update(aux_loss.item())

            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()

            # update the parameters of discriminator
            decompressed_xyzs_det = autograd.Variable(decompressed_xyzs.detach(), requires_grad=True)
            dis_optimizer.zero_grad()
            cd_dist = loss.detach()
            fake = discr(decompressed_xyzs_det)
            real = discr(input)
            dis_d_loss = -torch.pow(real-fake,2)
            dis_p_loss = args.grad_pen_coe*gradientPenalty(decompressed_xyzs_det,fake,cd_dist,args.K_coe)
            dis_all_loss = dis_d_loss+dis_p_loss
            epoch_dis_d_loss.update(dis_d_loss.item())
            epoch_dis_p_loss.update(dis_p_loss.item())
            
            dis_all_loss.backward()
            dis_optimizer.step()

        scheduler_steplr.step()

        # print loss
        interval = time.time() - start
        print("train epoch: %d/%d, time: %d mins %.1f secs, loss: %f, avg chamfer loss: %f, "
              "avg dis_g loss: %f, avg latent xyzs loss: %f, "
              "avg normal loss: %f, avg bpp loss: %f, avg aux loss: %f, " 
              "avg dis_d loss: %f, avg dis_p loss: %f" %
              (epoch+1, args.epochs, interval/60, interval%60, epoch_loss.get_avg(), epoch_chamfer_loss.get_avg(),
              epoch_dis_g_loss.get_avg(), epoch_latent_xyzs_loss.get_avg(),
              epoch_normal_loss.get_avg(), epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg(),
              epoch_dis_d_loss.get_avg(), epoch_dis_p_loss.get_avg()))
        


        # validation
        model.eval()
        val_chamfer_loss = AverageMeter()
        val_normal_loss = AverageMeter()
        val_bpp = AverageMeter()
        with torch.no_grad():
            for input_dict in val_loader:
                # xyzs: (b, n, c)
                input = input_dict['xyzs'].cuda()
                # (b, c, n)
                input = input.permute(0, 2, 1).contiguous()

                # compress normal
                if args.compress_normal == True:
                    normals = input_dict['normals'].cuda().permute(0, 2, 1).contiguous()
                    input = torch.cat((input, normals), dim=1)
                    args.in_fdim = 6

                # gt_xyzs
                gt_xyzs = input[:, :3, :].contiguous()

                # model forward
                decompressed_xyzs, loss, loss_items, bpp = model(input)
                # calculate val loss and bpp
                d1, d2, _, _ = chamfer_dist(gt_xyzs.permute(0, 2, 1).contiguous(),
                                            decompressed_xyzs.permute(0, 2, 1).contiguous())
                chamfer_loss = d1.mean() + d2.mean()
                val_chamfer_loss.update(chamfer_loss.item())
                val_normal_loss.update(loss_items['normal_loss'])
                val_bpp.update(bpp.item())

        # print loss
        print("val epoch: %d/%d, val bpp: %f, val chamfer loss: %f, val normal loss: %f" %
              (epoch+1, args.epochs, val_bpp.get_avg(), val_chamfer_loss.get_avg(), val_normal_loss.get_avg()))

        # save checkpoint
        cur_val_chamfer_loss = val_chamfer_loss.get_avg()
        if  cur_val_chamfer_loss < best_val_chamfer_loss or (epoch+1) % args.save_freq == 0:
            model_name = 'ckpt-best.pth' if cur_val_chamfer_loss < best_val_chamfer_loss else 'ckpt-epoch-%02d.pth' % (epoch+1)
            model_path = os.path.join(checkpoint_dir, model_name)
            
            state_dict = {'epoch': epoch,
              'best_chamfer_loss': best_val_chamfer_loss,
              'state_dict_g': model.state_dict(),
              'state_dict_d': discr.state_dict()}
            
            torch.save(state_dict, model_path)
            # update best val chamfer loss
            if cur_val_chamfer_loss < best_val_chamfer_loss:
                best_val_chamfer_loss = cur_val_chamfer_loss




def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
        setattr(model_args, arg, getattr(train_args, arg))




def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='semantickitti', type=str, help='shapenet or semantickitti')
    parser.add_argument('--model_path', default=None, type=str, help='path to ckpt')
    parser.add_argument('--downsample_rate', default=[1/3, 1/3, 1/3], nargs='+', type=float, help='downsample rate')
    parser.add_argument('--max_upsample_num', default=[8, 8, 8], nargs='+', type=int, help='max upsmaple number, reversely symmetric with downsample_rate')
    parser.add_argument('--bpp_lambda', default=1e-3, type=float, help='bpp loss coefficient')
    # normal compression
    parser.add_argument('--compress_normal', default=False, type=str2bool, help='whether compress normals')
    # compress latent xyzs
    parser.add_argument('--quantize_latent_xyzs', default=True, type=str2bool, help='whether compress latent xyzs')
    parser.add_argument('--latent_xyzs_conv_mode', default='mlp', type=str, help='latent xyzs conv mode, mlp or edge_conv')
    # sub_point_conv mode
    parser.add_argument('--sub_point_conv_mode', default='mlp', type=str, help='sub-point conv mode, mlp or edge_conv')
    parser.add_argument('--output_path', default='./output', type=str, help='output path')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['shapenet', 'semantickitti']

    if train_args.dataset == 'shapenet':
        model_args = parse_shapenet_args()
    else:
        model_args = parse_semantickitti_args()

    reset_model_args(train_args, model_args)

    train(model_args)
