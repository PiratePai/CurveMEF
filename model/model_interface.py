#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:model_interface.py
@time:2022/04/22
"""
import copy
import wandb
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision.utils as tutils
from pytorch_lightning.utilities import rank_zero_only
from torch.autograd import Variable
from torchvision import transforms

from data import tensor2save
from loss import get_loss, compute_loss
from model.choose_model import choose_model

to_pil = transforms.ToPILImage()


class MInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # 存储参数

        self.stage_ = None
        self.save_hyperparameters()
        self.cfg = args
        # 储存参数目录
        self.backcone = args.model
        self.weight_dirs = args.save_dir
        self.best_dirs = Path(self.weight_dirs).joinpath('best')
        self.log_intervals = args.log_intervals.num
        # 储存测试输出
        self.test_save_dir = Path(self.weight_dirs).joinpath('output')
        # 载入模型
        self.model = self.get_model()
        # 损失函数相关
        self.loss_term = copy.deepcopy(args.loss_term)
        self.loss_name, self.loss_model, self.loss_weight = get_loss(self.loss_term, args.device)
        # 配置optimizer
        self.optim_term = copy.deepcopy(args.optim_term)
        self.total_epochs = args.total_epochs
        self.epochs_gap = 1
        if self.epochs_gap == 0:
            self.epochs_gap = 1
        # 模型平均
        self.save_flag = 1000000
        self.save_hyperparameters()
        if 'y' in self.backcone.model_name:
            self.test_tensor = torch.load('test1pngy.pt')
        else:
            self.test_tensor = torch.load('test1png.pt')

    def forward(self, high_, low_):
        fused = self.model.forward(high_, low_)
        return fused

    @torch.no_grad()
    def predict(self, batch, batch_idx):
        high_, low_, high_path, low_path = batch
        high_ = Variable(high_, requires_grad=False)
        low_ = Variable(low_, requires_grad=False)
        fused_ = self.model.forward(high_, low_)
        return fused_

    def calculate_loss(self, high_, low_, fused_):
        loss_values_1, loss_items_1 = compute_loss(self.loss_model, self.loss_weight, high_, fused_)
        loss_values_2, loss_items_2 = compute_loss(self.loss_model, self.loss_weight, low_, fused_)
        # 列表逐项相加
        loss_values = []
        loss_items = []
        for idx_loss, it_loss in enumerate(loss_items_1):
            loss_values.append(loss_values_1[idx_loss] + loss_values_2[idx_loss])
            loss_items.append(it_loss + loss_items_2[idx_loss])
        return loss_values, loss_items

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: [torch.Size([batch_size, channel_num, crop_size, crop_size]),
                               [batch_size, channel_num, crop_size, crop_size])
            batch_idx: 序号
        Returns: loss 损失
        """
        self.stage_ = 'fit'
        high_, low_ = batch
        high_ = Variable(high_, requires_grad=False)
        low_ = Variable(low_, requires_grad=False)
        fused_ = self.model.forward(high_, low_)

        x = Variable(high_.data.clone(), requires_grad=False)
        y = Variable(low_.data.clone(), requires_grad=False)

        loss_values, loss_items = self.calculate_loss(x, y, fused_)

        total_loss = sum(loss_items)

        if self.global_step % self.log_intervals == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )

            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            # print(loss_items)
            for idx1, item1 in enumerate(loss_values):
                log_msg += "{}:{:.4f}| ".format(
                    self.loss_name[idx1], item1)
            for idx2, item2 in enumerate(loss_values):
                self.scalar_summary(
                    "Train / " + self.loss_name[idx2],
                    "Train", item2,
                    self.global_step, )
            self.logger.info(log_msg)
            for idx3, item3 in enumerate(loss_values):
                self.log('train_{}'.format(self.loss_name[idx3]), item3)
            self.log('train_loss', total_loss)
        return total_loss

    def training_epoch_end(self, outputs):
        self.trainer.save_checkpoint(Path(self.weight_dirs).joinpath("model_last.ckpt"))
        self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        self.stage_ = 'val'
        if self.global_step == 0:
            wandb.define_metric('val_loss', summary='min')
        high_, low_, = batch
        high_ = Variable(high_, requires_grad=False)
        low_ = Variable(low_, requires_grad=False)
        fused_ = self.model.forward(high_, low_)
        x = Variable(high_.data.clone(), requires_grad=False)
        y = Variable(low_.data.clone(), requires_grad=False)

        val_loss_values, val_loss_items = self.calculate_loss(x, y, fused_)

        val_total_loss = sum(val_loss_items)

        for vidx, vitem in enumerate(val_loss_values):
            self.log('val_{}'.format(self.loss_name[vidx]), vitem)
        self.log('val_loss', val_total_loss)

        if batch_idx % self.log_intervals == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            for vidx1, vitem1 in enumerate(val_loss_values):
                log_msg += "{}:{:.4f}| ".format(
                    self.loss_name[vidx1], vitem1)
            log_msg += "{}:{:.4f}| ".format(
                'Loss', val_total_loss)
            self.logger.info(log_msg)
        return val_loss_items, val_total_loss

    def validation_epoch_end(self, validation_step_outputs):
        [Losses, metric] = validation_step_outputs[0][0:4]

        if metric < self.save_flag:
            self.save_flag = metric
            best_save_path = Path(self.best_dirs).joinpath("model_best")
            self.trainer.save_checkpoint(
                Path(best_save_path).joinpath("model_best.ckpt")
            )
            self.save_model_state(
                Path(best_save_path).joinpath("CurveMEF_model_best.pth")
            )

            x_, y_ = self.test_tensor
            path_ = Path(Path(self.weight_dirs).joinpath('samples')).joinpath("Epoch_{}.png".format(self.current_epoch))
            with torch.no_grad():
                f_ = self.model.forward(x_, y_)
            tutils.save_image(f_, path_)

            if self.current_epoch % self.epochs_gap == 0:
                self.trainer.save_checkpoint(
                    Path(best_save_path).joinpath("model_best_epoch_{}.ckpt".format(self.current_epoch))
                )
            txt_path = Path(best_save_path).joinpath("eval_results.txt")
            if self.local_rank < 1:
                with open(txt_path, "a") as f:
                    f.write("Epoch:{}\n".format(self.current_epoch + 1))
                    for idx, item in enumerate(Losses):
                        f.write("{}: {}\n".format(self.loss_name[idx], item))
                    f.write("{}: {}\n".format('Loss', metric))

    def test_step(self, batch, batch_idx):
        fused_pic = self.predict(batch, batch_idx)
        high_, low_, high_path, low_path = batch
        # if self.model_name == 'ghost_curve_2iny':
        if 'y' in self.backcone.model_name:
            tensor2save(fused_pic, [high_path, low_path], self.test_save_dir, 1)
        else:
            tensor2save(fused_pic, [high_path, low_path], self.test_save_dir, 3)

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.optim_term.optimizer)
        op_name = optimizer_cfg.pop('name')
        build_optimizer = getattr(torch.optim, op_name)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)
        schedule_cfg = copy.deepcopy(self.optim_term.lr_schedule)
        ls_name = schedule_cfg.pop('name')
        build_scheduler = getattr(torch.optim.lr_scheduler, ls_name)
        self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        return optimizer

    def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_idx=None,
            optimizer_closure=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.optim_term.warmup.steps:
            if self.optim_term.warmup.name == "constant":
                warmup_lr = (
                        self.optim_term.optimizer.lr * self.optim_term.warmup.ratio
                )
            elif self.optim_term.warmup.name == "linear":
                k = (1 - self.trainer.global_step / self.optim_term.warmup.steps) * (
                        1 - self.optim_term.warmup.ratio
                )
                warmup_lr = self.optim_term.optimizer.lr * (1 - k)
            elif self.optim_term.warmup.name == "exp":
                k = self.optim_term.warmup.ratio ** (
                        1 - self.trainer.global_step / self.optim_term.warmup.steps
                )
                warmup_lr = self.optim_term.optimizer.lr * k
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # update params

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    # ------------Hooks-----------------
    def on_train_start(self) -> None:
        if self.current_epoch > 0:
            self.lr_scheduler.last_epoch = self.current_epoch - 1

    def get_model(self):
        net_model = choose_model(self.backcone.model_name)
        return net_model
