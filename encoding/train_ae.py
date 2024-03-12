from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_builder import dataset_builder
from encoding.networks import AutoEncoderGroupSkip
from encoding.lovasz import lovasz_softmax
from utils.utils import save_remap_lut, point2voxel
import os
import torch
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from encoding.ssc_metrics import SSCMetrics

class Trainer:
    def __init__(self, args):
        # etc
        self.args = args
        self.writer = SummaryWriter(os.path.join(args.save_path, 'tb'))
        self.epoch, self.start_epoch = 0, 0
        self.global_step = 0
        self.best_miou = 0

        # dataset
        self.train_dataset, self.val_dataset, self.num_class, class_names = dataset_builder(args)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self.iou_class_names = class_names

        # model & optimizer
        self.model = AutoEncoderGroupSkip(args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.lr_scheduler_steps, args.lr_scheduler_decay) if args.lr_scheduler else None
        self.grad_scaler = GradScaler()
        
        if args.resume:
            checkpoint = torch.load(args.resume)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
            # TODO: load scheduler

        # loss functions
        self.loss_fns = {}
        self.loss_fns['ce'] = torch.nn.CrossEntropyLoss(weight=self.train_dataset.weights, ignore_index=255)
        self.loss_fns['lovasz'] = None

    def train(self):
        for epoch in range(30000):
            self.epoch = self.start_epoch + epoch + 1
                
            print('Training...')
            self._train_model()
            
            if epoch % self.args.eval_epoch == 0:
                print('Evaluation...')
                self._eval_and_save_model()

            # learning rate scheduling
            self.scheduler.step()
            self.writer.add_scalar('lr_epochwise', self.optimizer.param_groups[0]['lr'], global_step=self.epoch)

    def _loss(self, vox, query, label, losses, coord):
        empty_label = 0.
        preds = self.model(vox, query) # [bs, N, 20]
        losses['ce'] = self.loss_fns['ce'](preds.view(-1, self.num_class), label.view(-1,))
        losses['loss'] = losses['ce']
        
        pred_output = torch.full((preds.shape[0], vox.shape[1], vox.shape[2], vox.shape[3], self.num_class), fill_value=empty_label, device=preds.device)
        gt_output = torch.full((preds.shape[0], vox.shape[1], vox.shape[2], vox.shape[3]), fill_value=empty_label, device=preds.device)
        softmax_preds = torch.nn.functional.softmax(preds, dim=2)
        for i in range(softmax_preds.shape[0]):
            pred_output[i, coord[i, :, 0], coord[i, :, 1], coord[i, :, 2], :] = softmax_preds[i]
            gt_output[i, coord[i, :, 0], coord[i, :, 1], coord[i, :, 2]] = label[i].float()
        losses['lovasz'] = lovasz_softmax(pred_output.permute(0,4,1,2,3), gt_output)
        losses['loss'] += losses['lovasz']

        adaptive_weight = None
        return losses, preds, adaptive_weight
    
    def _train_model(self):
        self.model.train()

        total_losses = {loss_name: 0. for loss_name in self.loss_fns.keys()}
        total_losses['loss'] = 0.
        evaluator = SSCMetrics(self.num_class, [])
        dataloader_tqdm = tqdm(self.train_dataloader)

        for vox, query, label, coord, path, invalid in dataloader_tqdm:
            vox = vox.type(torch.LongTensor).cuda()
            query = query.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            coord = coord.type(torch.LongTensor).cuda()
            invalid = invalid.type(torch.LongTensor).cuda()
            b_size = vox.size(0)  # TODO: bsize is correct?

            # forward
            losses = {}
            with autocast():
                losses, model_output, adaptive_weight = self._loss(vox, query, label, losses, coord)

            # optimize
            self.optimizer.zero_grad()
            self.grad_scaler.scale(losses['loss']).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # eval and log each iteration
            if self.global_step % self.args.display_period == 0:
                pred_mask = get_pred_mask(model_output)

                masks = torch.from_numpy(evaluator.get_eval_mask(vox.cpu().numpy(), invalid.cpu().numpy()))
                output = point2voxel(self.args, pred_mask, coord)
                eval_output = output[masks]
                eval_label = vox[masks]
                this_iou, this_miou = evaluator.addBatch(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))

                # on display
                dataloader_tqdm.set_postfix({"loss": losses['loss'].detach().item(), "iou": this_iou, "miou": this_miou})

                # on tensorboard
                self.writer.add_scalar('Grad_Norm', grad_norm, global_step=self.global_step)
                self.writer.add_scalar('Train_Performance_stepwise/IoU', this_iou, global_step=self.global_step)
                self.writer.add_scalar('Train_Performance_stepwise/mIoU', this_miou, global_step=self.global_step)
                for loss_name in losses.keys():
                    self.writer.add_scalar(f'Train_Loss_stepwise/loss_{loss_name}', losses[loss_name], self.global_step)
          
            # loss accumulation for logging
            for loss_name in losses.keys():
                total_losses[loss_name] += (losses[loss_name] * b_size)

            self.global_step += 1

        # eval for 1 epoch
        _, class_jaccard = evaluator.getIoU()
        m_jaccard = class_jaccard[1:].mean()
        miou = m_jaccard * 100
        conf = evaluator.get_confusion()
        iou = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0] + 1e-8)
        evaluator.reset()

        # log for 1 epoch
        self.writer.add_scalar('Train_Performance_epochwise/IoU', iou, global_step=self.epoch)
        self.writer.add_scalar('Train_Performance_epochwise/mIoU', miou, global_step=self.epoch)
        for class_idx, class_name in enumerate(self.iou_class_names):
            self.writer.add_scalar(f'Train_ClassPerformance_epochwise/class{class_idx + 1}_IoU_{class_name}', class_jaccard[class_idx + 1], global_step=self.epoch)
        for loss_name in losses.keys():
            self.writer.add_scalar(f'Train_Loss_epochwise/loss_{loss_name}', total_losses[loss_name] / len(self.train_dataset), global_step=self.epoch)

        print(f"Epoch: {self.epoch} \t IOU: \t {iou:01f} \t mIoU: \t {miou:01f}")


    @torch.no_grad()
    def _eval_and_save_model(self):
        self.model.eval()

        total_losses = {loss_name: 0. for loss_name in self.loss_fns.keys()}
        total_losses['loss'] = 0.
        evaluator = SSCMetrics(self.num_class, [])
        dataloader_tqdm = tqdm(self.val_dataloader)

        for sample_idx, (vox, query, label, coord, path, invalid) in enumerate(dataloader_tqdm):
            vox = vox.type(torch.LongTensor).cuda()
            query = query.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            coord = coord.type(torch.LongTensor).cuda()
            invalid = invalid.type(torch.LongTensor).cuda()
            b_size = vox.size(0)  # TODO: check correctness
            assert b_size == 1, 'For accurate logging, please set batch size of validation dataloader to 1.'

            losses = {}
            losses, model_output, adaptive_weight = self._loss(vox, query, label, losses, coord)
            pred_mask =  get_pred_mask(model_output)

            masks = torch.from_numpy(evaluator.get_eval_mask(vox.cpu().numpy(), invalid.cpu().numpy()))
            output = point2voxel(self.args, pred_mask, coord)
            eval_output = output[masks]
            eval_label = vox[masks]
            this_iou, this_miou = evaluator.addBatch(eval_output.cpu().numpy().astype(int), eval_label.cpu().numpy().astype(int))

            # log on display for each sample
            dataloader_tqdm.set_postfix({"loss": losses['loss'].detach().item(), "iou": this_iou, "miou": this_miou})

            for loss_name in losses.keys():
                total_losses[loss_name] += (losses[loss_name] * b_size)

            idx = path[0].split('/')[-1].split('.')[0]
            folder = path[0].split('/')[-3]
            save_remap_lut(self.args, output, folder, idx, self.train_dataset.learning_map_inv, True)

        # eval for all validation samples
        _, class_jaccard = evaluator.getIoU()
        m_jaccard = class_jaccard[1:].mean()
        miou = m_jaccard * 100
        conf = evaluator.get_confusion()
        iou = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0] + 1e-8)
        evaluator.reset()

        self.writer.add_scalar('Val_Performance_epochwise/IoU', iou, global_step=self.epoch)
        self.writer.add_scalar('Val_Performance_epochwise/mIoU', miou, global_step=self.epoch)
        for class_idx, class_name in enumerate(self.iou_class_names):
            self.writer.add_scalar(f'Val_ClassPerformance_epochwise/class{class_idx + 1}_IoU_{class_name}', class_jaccard[class_idx + 1], global_step=self.epoch)
        for loss_name in losses.keys():
            self.writer.add_scalar(f'Val_Loss_epochwise/loss_{loss_name}', total_losses[loss_name] / len(self.val_dataset), global_step=self.epoch)
        print(f"Epoch: {self.epoch} \t IOU: \t {iou:01f} \t mIoU: \t {miou:01f}")

        if self.best_miou < miou:
            self.best_miou = miou
            checkpoint = {'optimizer': self.optimizer.state_dict(), 'model': self.model.state_dict(), 'epoch': self.epoch}  # TODO: save scheduler
            torch.save(checkpoint, self.args.save_path + "/" + str(self.epoch) + "_miou=" + str(f"{miou:.3f}") + '.pt')

def get_pred_mask(model_output, separate_decoder=False):
    preds = model_output
    pred_prob = torch.softmax(preds, dim=2)
    pred_mask = pred_prob.argmax(dim=2).float()
    return pred_mask
