from absl import flags, app
from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored
from tqdm import trange, tqdm

from nnutils import resnet, xception, ttn, ccr_nets
from nnutils import loss_utils
from nnutils import task_decoders as decoders
from nnutils import depth_decoders

from data.idd_unsupervised_mixed import TrainData, ValData

# UM Adapt Parameters
flags.DEFINE_float('lmbda', 10.0, 'Lambda parameter of UM-Adapt, Algorithm 2')
flags.DEFINE_string('backbone', 'resnet', 'Which backbone should the model use? [resnet, xception]')
flags.DEFINE_string('depth_decoder', 'bts', 'Which depth decoder is to be used? [bts, fcrnd]')

# Loss Specific
#    -> Weights
flags.DEFINE_float('seg_loss_wt', 1.0, 'Weight of the semantic segmentation loss')
flags.DEFINE_float('ins_loss_wt', 1.0, 'Weight of the overall instance segmentation loss')
flags.DEFINE_float('dep_loss_wt', 1.0, 'Weight of the depth loss')
flags.DEFINE_float('ins_center_wt', 200.0, 'Weight of the instance center loss that contributes to the overall instance loss')
flags.DEFINE_float('ins_offset_wt', 0.01, 'Weight of the instance offset loss that contributes to the overall instance loss')
#    -> Misc
flags.DEFINE_float('top_k_percent_pixels', 0.2, 'Top K percentage pixels for deeplab cross entropy loss')
flags.DEFINE_integer('ignore_label', 19, 'Ignore this label for cross entropy loss')
flags.DEFINE_boolean('use_smooth_l1', False, 'Use smooth l1 loss for instace offsets')

# Data Specific
flags.DEFINE_integer('batch_size', 16, 'Batch Size of training')
flags.DEFINE_integer('val_batch_size', 32, 'Batch Size while validating')
flags.DEFINE_integer('data_threads', 16, 'Threads the dataloader is allowed to use')
flags.DEFINE_boolean('shuffle_data', True, 'Shuffle data while training')
flags.DEFINE_boolean('pin_memory', True, 'Set pin_memory while training')
flags.DEFINE_boolean('drop_last', True, 'Drop the last incomplete batch while training')

flags.DEFINE_integer('image_width', 512, 'Width of the input image. Dataloader will resize all images to this size')
flags.DEFINE_integer('image_height', 0, 'Height of the input image. If 0, it will be taken as width/2')
flags.DEFINE_integer('val_image_width', 0, 'Width of the validation image. If 0, it will be taken as image_width')
flags.DEFINE_integer('val_image_height', 0, 'Height of the validation image. If 0, it will be taken as val_image_width/2')

# Training Specific
flags.DEFINE_string('name', None, 'The name of this run')
flags.DEFINE_string('save_location', 'cachedir', 'Location where to store the trained models & log files')
flags.DEFINE_integer('val_freq', 1, 'How frequently should the training be validated?')
flags.DEFINE_integer('epochs', 100, 'How many epochs to train for?')
flags.DEFINE_integer('save_freq', 50, 'How frequently should the model be saved?')
flags.DEFINE_integer('save_latest_freq', 10, 'How frequently should the latest model be saved? [This will overwrite previous latest]')
flags.DEFINE_boolean('use_amp', False, 'Use AMP while training')
flags.DEFINE_boolean('multi_gpu', False, 'Use DataParallel and train across multiple GPUs')

flags.DEFINE_float('lr', 0.0001, 'Learning Rate')
flags.DEFINE_float('lr_factor', 0.1, 'Decay Learning Rate by this factor every time the validation plateaus')
flags.DEFINE_integer('lr_cooldown', 3, 'Cooldown')
flags.DEFINE_integer('lr_patience', 10, 'Patience')

flags.DEFINE_string('optimizer', 'adam', 'choices are [adam, sgd]')
flags.DEFINE_boolean('viz_epochs', True, 'Have one bar visualizing the training')
flags.DEFINE_boolean('viz_each_epoch', False, 'Have one bar for each epoch')

flags.DEFINE_string('base_name', None, 'Name of the base model to be loaded')
flags.DEFINE_string('base_epoch', 'best_miou', 'Which epoch of the base model to load?')

# Model related
flags.DEFINE_integer('inst_channels', 128, 'number of channels in the output of the instance segmentation network')
flags.DEFINE_integer('num_classes', 20, 'number of classes for semantic segmentation')
flags.DEFINE_boolean('ccr_with_bn', False, 'Use BatchNorm in CCR Net.')
flags.DEFINE_string('hed_path', None, 'Path to the pretrained HED Network.')

# Logging
flags.DEFINE_boolean('use_tensorboard', True, 'Use tensorboard to log losses')

# Set Required Flags
flags.mark_flag_as_required('name')
#flags.mark_flag_as_required('image_width')
#flags.mark_flag_as_required('batch_size')
#flags.mark_flag_as_required('data_threads')
#flags.mark_flag_as_required('epochs')
#flags.mark_flag_as_required('lr')

opts = flags.FLAGS

class UMATrainer:
    def __init__(self, opts):
        self.opts = opts
        opts_dir = os.path.join(opts.save_location, 'opts')
        if not os.path.isdir(opts_dir):
            os.mkdir(opts_dir)
        opts_path = os.path.join(opts_dir, opts.name)
        if not os.path.isdir(opts_path):
            os.mkdir(opts_path)
        log_file = os.path.join(opts_path, 'opts.log')
        with open(log_file, 'w') as f:
            for k in dir(opts):
                f.write('{}: {}\n'.format(k, opts.__getattr__(k)))
    
    def load_models(self):
        # Encoders
        #Base Encoders
        if opts.backbone == 'xception':
            self.backbone.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/xcep_{opts.base_epoch}.pth'))
        else:
            self.base_resnet4f.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/r4_{opts.base_epoch}.pth'))
            self.base_resnet5.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/r5_{opts.base_epoch}.pth'))
        # Task Specific Decoders
        self.seg_decoder.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/segd_{opts.base_epoch}.pth'))
        self.ins_decoder.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/insd_{opts.base_epoch}.pth'))
        self.dep_decoder.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/depd_{opts.base_epoch}.pth'))
        # Task Transfer Networks
        self.ttn_id2s.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/tid2s_{opts.base_epoch}.pth'))
        self.ttn_is2d.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/tis2d_{opts.base_epoch}.pth'))
        self.ttn_sd2i.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}/tsd2i_{opts.base_epoch}.pth'))
        #CCR Nets
        self.ccr.load_state_dict(torch.load(f'cachedir/models/{opts.base_name}_ccr/ccr_nn_best_val.pth'))
        self.hed_net.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(opts.hed_path).items()})
        print(colored('Loaded Pretrained Models', 'green'))
    
    def define_models(self):
        opts = self.opts
        # Encoders
        if opts.backbone == 'xception':
            self.backbone = xception.xception65(pretrained=True).cuda()
        else:
            self.base_resnet4f = resnet.ResnetBackbone().cuda()
            self.base_resnet5 = resnet.ResnetBackbone5().cuda()
        
        # Task Specific Decoders
        self.seg_decoder = decoders.PDLSegDecoder(opts.num_classes, low_level_channel=self.low_level_channel).cuda()
        self.ins_decoder = decoders.PDLInstDecoder(low_level_channel=self.low_level_channel).cuda()

        if opts.depth_decoder == 'bts':
            self.dep_decoder = depth_decoders.BTS(opts, self.depth_channels).cuda()
        else:
            self.dep_decoder = depth_decoders.FCRNDepthDecoder(output_size=(opts.image_height, opts.image_width)).cuda()
        
        # Task Transfer Networks
        self.ttn_id2s = ttn.TTNInst_Dep2Seg(opts.inst_channels).cuda()
        self.ttn_is2d = ttn.TTNInst_Seg2Dep(opts.inst_channels, opts.num_classes).cuda()
        self.ttn_sd2i = ttn.TTNSeg_Dep2Inst(opts.num_classes).cuda()
        
        # CCR Networks
        self.ccr = ccr_nets.CCRNet(with_bn=opts.ccr_with_bn).cuda()
        self.hed_net = ccr_nets.HED_Network(output_shape=(opts.image_height // 2, opts.image_width // 2)).cuda()

        self.load_models()
        
        if opts.multi_gpu:
            print(colored('Using multi-GPU training...', 'yellow'))
            self.base_resnet4f = nn.DataParallel(self.base_resnet4f)
            self.base_resnet5 = nn.DataParallel(self.base_resnet5)
            self.seg_decoder = nn.DataParallel(self.seg_decoder)
            self.ins_decoder = nn.DataParallel(self.ins_decoder)
            self.dep_decoder = nn.DataParallel(self.dep_decoder)
            self.ttn_id2s = nn.DataParallel(self.ttn_id2s)
            self.ttn_is2d = nn.DataParallel(self.ttn_is2d)
            self.ttn_sd2i = nn.DataParallel(self.ttn_sd2i)
            self.ccr = nn.DataParallel(self.ccr)
            self.hed_net = nn.DataParallel(self.hed_net)
    
    def define_losses(self):
        opts = self.opts
        self.seg_loss = loss_utils.DeepLabCE(ignore_label=opts.ignore_label, top_k_percent_pixels=opts.top_k_percent_pixels)
        self.dep_loss = loss_utils.MaskedL1Loss()
        self.inst_center_loss = nn.MSELoss()
        if opts.use_smooth_l1:
            self.inst_offset_loss = nn.SmoothL1Loss()
        else:
            self.inst_offset_loss = nn.L1Loss()
    
    def init_optimizer(self):
        opts = self.opts
        self.train_params_base = self.base_resnet5.parameters()
        
        if opts.optimizer == 'adam':
            self.optimizer = optim.AdamW(self.train_params_base, lr=opts.lr)
        else:
            print('Using SGD optimizer')
            self.optimizer = optim.SGD(self.train_params_base, lr=opts.lr)
            
        self.scaler = torch.cuda.amp.GradScaler(enabled=opts.use_amp)
    
    def init_dataset(self):
        opts = self.opts
        self.train_data = TrainData(opts)
        self.val_data = ValData(opts)
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=opts.batch_size, shuffle=opts.shuffle_data, num_workers=opts.data_threads, pin_memory=opts.pin_memory, drop_last=opts.drop_last)
        self.valloader = torch.utils.data.DataLoader(self.val_data, batch_size=opts.val_batch_size, shuffle=opts.shuffle_data, num_workers=opts.data_threads, pin_memory=opts.pin_memory)
        
    def init_logger(self):
        opts = self.opts
        if opts.use_tensorboard:
            log_path = os.path.join(opts.save_location, 'tb_runs')
            if not os.path.isdir(log_path):
                os.mkdir(log_path)
            now = datetime.now()
            writer_path = os.path.join(log_path, opts.name + '_' + now.strftime("%d%m%Y-%H%M%S"))
            self.writer = SummaryWriter(writer_path)
    
    def init_images(self):
        opts = self.opts
        if opts.image_height == 0:
            opts.image_height = opts.image_width // 2
        if opts.val_image_width == 0:
            opts.val_image_width = opts.image_width
        if opts.val_image_height == 0:
            opts.val_image_height = opts.val_image_width // 2

        print(f'Input Image Size : ({opts.image_height}, {opts.image_width})')
        print(f'Val Image Size   : ({opts.val_image_height}, {opts.val_image_width})')

        if opts.backbone == 'xception':
            self.low_level_channel=(728, 728, 256)
            self.depth_channels=(64, 256, 728, 728, 2048)
        else:
            self.low_level_channel=(1024, 512, 256)
            self.depth_channels=(64, 256, 512, 1024, 2048)

    def init_training(self):
        self.init_images()
        self.define_models()
        self.define_losses()
        self.init_optimizer()
        self.init_dataset()
        self.init_logger()
        self.best_val = 0.0
        print(colored('Initialization Successful', 'green'))
    
    def save(self, epoch_label):
        save_path_base = os.path.join(opts.save_location, 'models')
        if not os.path.isdir(save_path_base):
            os.mkdir(save_path_base)
        save_path = os.path.join(save_path_base, opts.name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        if opts.multi_gpu:
            #Base Encoders
            torch.save(self.base_resnet5.module.state_dict(), save_path + f'/r4_{epoch_label}.pth')
        else:
            #Base Encoders
            torch.save(self.base_resnet5.state_dict(), save_path + f'/r5_{epoch_label}.pth')
        
    def write_logs(self, seg_loss, ins_loss, dep_loss, ccr_loss, log_prefix):
        self.writer.add_scalar(f'Loss/{log_prefix}_seg', seg_loss, self.curr_epoch)
        self.writer.add_scalar(f'Loss/{log_prefix}_inst', ins_loss, self.curr_epoch)
        self.writer.add_scalar(f'Loss/{log_prefix}_dep', dep_loss, self.curr_epoch)
        self.writer.add_scalar(f'Loss/{log_prefix}_ccr', ccr_loss, self.curr_epoch)
        self.writer.flush()
    
    @staticmethod
    def compute_metrics(conf_matrix, num_classes=19):
        acc = np.zeros(num_classes, dtype=np.float)
        iou = np.zeros(num_classes, dtype=np.float)
        tp = conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = miou
        res["fwIoU"] = fiou
        res["mACC"] = macc
        res["pACC"] = pacc

        return res
    
    def validate(self):
        val_offsets_base = 0.0
        val_offsets_ttn = 0.0
        N = self.opts.num_classes
        conf_matrix = np.zeros((N, N), dtype=np.int64)
        conf_matrix_ttn = np.zeros((N, N), dtype=np.int64)

        with torch.cuda.amp.autocast(enabled=opts.use_amp):
            with torch.no_grad():
                for data_batch in self.valloader:
                    if opts.backbone == 'xception':
                        metadata = self.backbone(data_batch['image'].float().cuda())
                        r5_feats = metadata['res5']
                    else:
                        r4_feats, metadata = self.base_resnet4f(data_batch['image'].float().cuda())
                        r5_feats = self.base_resnet5(r4_feats)
                        metadata['res5'] = r5_feats
                    
                    segs = self.seg_decoder(r5_feats, metadata)['semseg']
                    insts = self.ins_decoder(r5_feats, metadata)
                    deps = self.dep_decoder(metadata)
                    ttn_insts = self.ttn_sd2i(segs, deps)
                    ttn_segs = self.ttn_id2s(insts, deps)
                    ttn_deps = self.ttn_is2d(insts, segs)

                    gt_segs = data_batch['semseg'].long().cpu().numpy()
                    gt_inst_o = data_batch['inst-offsets'].float().cuda()
                    
                    # Segmentation Losses
                    pred_seg = torch.argmax(F.softmax(segs, 1), 1).cpu().numpy()
                    conf_matrix += np.bincount(N * pred_seg.reshape(-1) + gt_segs.reshape(-1), minlength=N**2).reshape(N, N)
                    pred_seg_ttn = torch.argmax(F.softmax(ttn_segs, 1), 1).cpu().numpy()
                    conf_matrix_ttn += np.bincount(N * pred_seg_ttn.reshape(-1) + gt_segs.reshape(-1), minlength=N**2).reshape(N, N)
                    
                    # Instance Loss
                    val_offsets_base += F.l1_loss(insts['offsets'], gt_inst_o).item()
                    val_offsets_ttn += F.l1_loss(ttn_insts['offsets'], gt_inst_o).item()
                    
        val_offsets_base /= len(self.valloader)
        val_offsets_ttn /= len(self.valloader)
        
        metrics = self.compute_metrics(conf_matrix)
        metrics_ttn = self.compute_metrics(conf_matrix_ttn)

        if opts.use_tensorboard:
            self.writer.add_scalar(f'Val/mIoU_base', metrics["mIoU"], self.curr_epoch)
            self.writer.add_scalar(f'Val/mIoU_ttn', metrics_ttn["mIoU"], self.curr_epoch)
            self.writer.add_scalar(f'Val/offset_err_base', val_offsets_base, self.curr_epoch)
            self.writer.add_scalar(f'Val/offset_err_ttn', val_offsets_ttn, self.curr_epoch)
            self.writer.flush()
        
    def train(self):
        opts = self.opts
        print(colored('Starting the training!', 'yellow'))
        for epoch in trange(opts.epochs, disable=not opts.viz_epochs):
            train_loss_seg = 0.0
            train_loss_inst = 0.0
            train_loss_dep = 0.0
            train_loss_ccr = 0.0
            self.curr_epoch = epoch + 1
            for data_batch in tqdm(self.trainloader, disable=not opts.viz_epochs):
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=opts.use_amp):
                    with torch.no_grad():
                        hed_gt = self.hed_net(data_batch['hed_image'].float().cuda())
                        r4_feats, metadata = self.base_resnet4f(data_batch['image'].float().cuda())

                    r5_feats = self.base_resnet5(r4_feats)
                    metadata['res5'] = r5_feats

                    ccr_preds = self.ccr(r5_feats)

                    segs = self.seg_decoder(r5_feats, metadata)['semseg']
                    insts = self.ins_decoder(r5_feats, metadata)
                    deps = self.dep_decoder(metadata)
                    
                    with torch.no_grad():
                        ttn_insts = self.ttn_sd2i(segs, deps)
                        ttn_segs = self.ttn_id2s(insts, deps)
                        ttn_deps = self.ttn_is2d(insts, segs)

                    # CCR Losses
                    loss_ccr = F.mse_loss(ccr_preds, hed_gt)
                    self.total_loss = opts.lmbda * loss_ccr
                    train_loss_ccr += loss_ccr.item()

                    # Segmentation Losses
                    pred_seg_ttn = torch.argmax(F.softmax(ttn_segs, 1), 1).long().detach()
                    loss_seg = self.seg_loss(segs, pred_seg_ttn)
                    self.total_loss += opts.seg_loss_wt * loss_seg
                    train_loss_seg += loss_seg.item()

                    # Instance Loss
                    loss_inst_c = self.inst_center_loss(insts['centers'], ttn_insts['centers'].detach())
                    loss_inst_o = self.inst_offset_loss(insts['offsets'], ttn_insts['offsets'].detach())
                    loss_inst = (opts.ins_center_wt * loss_inst_c) + (opts.ins_offset_wt * loss_inst_o)
                    self.total_loss += opts.ins_loss_wt * loss_inst
                    train_loss_inst += loss_inst.item()

                    # Depth Loss
                    ttn_dep_mask = ttn_deps > 0 
                    loss_dep = self.dep_loss(deps, ttn_deps.detach(), ttn_dep_mask.detach())
                    self.total_loss += opts.dep_loss_wt * loss_dep
                    train_loss_dep += loss_dep.item()
                    
                # Backward Pass & Update Optimizer
                self.scaler.scale(self.total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            train_loss_seg /= len(self.trainloader)
            train_loss_inst /= len(self.trainloader)
            train_loss_dep /= len(self.trainloader)
            train_loss_ccr /= len(self.trainloader)

            if opts.use_tensorboard:
                self.write_logs(train_loss_seg, train_loss_inst, train_loss_dep, train_loss_ccr, 'train')
            
            if self.curr_epoch % opts.val_freq == 0:
                self.validate()
            
            if self.curr_epoch % opts.save_freq == 0:
                self.save(self.curr_epoch)
                self.save('latest')
            
            if self.curr_epoch % opts.save_latest_freq == 0:
                self.save('latest')
            
        
        print(colored('Training Completed!', 'green'))
        if opts.use_tensorboard:
            self.writer.close()


def main(_):
    torch.manual_seed(0)
    trainer = UMATrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
