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

from nnutils import resnet, xception
from nnutils import ccr_nets

from data.cityscapes_ccr import TrainData, ValData

# UM Adapt Parameters
flags.DEFINE_string('backbone', 'resnet', 'Which backbone should the model use? [resnet, xception]')

# Loss Specific
#    -> Misc
flags.DEFINE_boolean('use_l1', False, 'Use l1 loss for CCR-Net training, default is MSE Loss.')

# Data Specific
flags.DEFINE_integer('batch_size', 16, 'Batch Size of training')
flags.DEFINE_integer('val_batch_size', 32, 'Batch Size while validating')
flags.DEFINE_integer('data_threads', 8, 'Threads the dataloader is allowed to use')
flags.DEFINE_boolean('shuffle_data', True, 'Shuffle data while training')
flags.DEFINE_boolean('pin_memory', True, 'Set pin_memory while training')
flags.DEFINE_boolean('drop_last', True, 'Drop the last incomplete batch while training')

flags.DEFINE_integer('image_width', 512, 'Width of the input image. Dataloader will resize all images to this size')
flags.DEFINE_integer('image_height', 0, 'Height of the input image. If 0, it will be taken as width/2')
flags.DEFINE_integer('val_image_width', 0, 'Width of the validation image. If 0, it will be taken as image_width')
flags.DEFINE_integer('val_image_height', 0, 'Height of the validation image. If 0, it will be taken as val_image_width/2')

# Training Specific
flags.DEFINE_string('save_location', 'cachedir', 'Location where to store the trained models & log files')
flags.DEFINE_integer('val_freq', 1, 'How frequently should the training be validated?')
flags.DEFINE_integer('epochs', 100, 'How many epochs to train for?')
flags.DEFINE_integer('save_freq', 50, 'How frequently should the model be saved?')
flags.DEFINE_integer('save_latest_freq', 10, 'How frequently should the latest model be saved? [This will overwrite previous latest]')
flags.DEFINE_boolean('use_amp', False, 'Use AMP while training')

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
flags.DEFINE_boolean('ccr_with_bn', False, 'Use BatchNorm in CCR Net.')
flags.DEFINE_string('hed_path', None, 'Path to the pretrained HED Network.')

# Logging
flags.DEFINE_boolean('use_tensorboard', True, 'Use tensorboard to log losses')

# Set Required Flags
flags.mark_flag_as_required('base_name')
flags.mark_flag_as_required('hed_path')
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
        opts_path = os.path.join(opts_dir, opts.base_name)
        if not os.path.isdir(opts_path):
            os.mkdir(opts_path)
        log_file = os.path.join(opts_path, 'opts_ccr.log')
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
        
        self.ccr = ccr_nets.CCRNet(with_bn=opts.ccr_with_bn).cuda()
        self.hed_net = ccr_nets.HED_Network(output_shape=(opts.image_height // 2, opts.image_width // 2)).cuda()
        self.load_models()

    def define_losses(self):
        opts = self.opts
        if opts.use_l1:
            self.ccr_loss = nn.L1Loss()
        else:
            self.ccr_loss = nn.MSELoss()
    
    def init_optimizer(self):
        opts = self.opts
        self.train_params = [{'params': self.ccr.parameters(), 'lr': opts.lr}]
        
        if opts.optimizer == 'adam':
            self.optimizer = optim.AdamW(self.train_params, lr=opts.lr)
        else:
            print('Using SGD optimizer')
            self.optimizer = optim.SGD(self.train_params, lr=opts.lr)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=opts.lr_factor, patience=opts.lr_patience, cooldown=opts.lr_cooldown)
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
            writer_path = os.path.join(log_path, 'ccr_' + opts.base_name + '_' + now.strftime("%d%m%Y-%H%M%S"))
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

    def init_training(self):
        self.init_images()
        self.define_models()
        self.define_losses()
        self.init_optimizer()
        self.init_dataset()
        self.init_logger()
        self.best_val = np.inf
        print(colored('Initialization Successful', 'green'))
    
    def save(self, epoch_label):
        save_path_base = os.path.join(opts.save_location, 'models')
        if not os.path.isdir(save_path_base):
            os.mkdir(save_path_base)
        save_path = os.path.join(save_path_base, opts.base_name + '_ccr')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        torch.save(self.ccr.state_dict(), save_path + f'/ccr_nn_{epoch_label}.pth')
        
    def write_logs(self, ccr_loss, log_prefix):
        self.writer.add_scalar(f'Loss/{log_prefix}_ccr', ccr_loss, self.curr_epoch)
        self.writer.flush()
    
    def validate(self):
        val_ccr_loss = 0.0

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
                    
                    ccr_preds = self.ccr(r5_feats)
                    hed_gt = self.hed_net(data_batch['hed_image'].float().cuda())
                    
                    val_ccr_loss += self.ccr_loss(ccr_preds, hed_gt).item()
                    
        val_ccr_loss /= len(self.valloader)
        
        self.scheduler.step(val_ccr_loss)

        if opts.use_tensorboard:
            self.write_logs(val_ccr_loss, 'val')
        
        if val_ccr_loss < self.best_val:
            self.best_val = val_ccr_loss
            self.save('best_val')

    def train(self):
        opts = self.opts
        print(colored('Starting the training!', 'yellow'))
        for epoch in trange(opts.epochs, disable=not opts.viz_epochs):
            train_loss_ccr = 0.
            self.curr_epoch = epoch + 1
            for data_batch in tqdm(self.trainloader, disable=not opts.viz_epochs):
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=opts.use_amp):
                    with torch.no_grad():
                        hed_gt = self.hed_net(data_batch['hed_image'].float().cuda())
                    
                        if opts.backbone == 'xception':
                            metadata = self.backbone(data_batch['image'].float().cuda())
                            r5_feats = metadata['res5']
                        else:
                            r4_feats, metadata = self.base_resnet4f(data_batch['image'].float().cuda())
                            r5_feats = self.base_resnet5(r4_feats)
                            metadata['res5'] = r5_feats
                        
                    ccr_preds = self.ccr(r5_feats)
                    self.total_loss = self.ccr_loss(ccr_preds, hed_gt)
                    train_loss_ccr += self.total_loss.item()

                # Backward Pass & Update Optimizer
                self.scaler.scale(self.total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            train_loss_ccr /= len(self.trainloader)
            
            if opts.use_tensorboard:
                self.write_logs(train_loss_ccr, 'train')
            
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
