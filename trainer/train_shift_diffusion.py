import argparse
import os
import copy
import time
from collections import defaultdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam

import model.denoise_fn as denoise_fn_module
import model.shift_predictor as shift_predictor_module
from diffusion.gaussian_diffusion import GaussianDiffusion

from utils.utils import  move_to_cuda, save_image
from trainer.base_trainer import BaseTrainer


class ShiftDiffusionTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: trainer initialized.'.format(self.global_rank))

    def _build_model(self):
        self.gaussian_diffusion = GaussianDiffusion(self.config["diffusion_config"], device=self.device)

        denoise_fn = getattr(denoise_fn_module, self.config["denoise_fn_config"]["model"], None)(**self.config["denoise_fn_config"])
        self.denoise_fn = DistributedDataParallel(copy.deepcopy(denoise_fn).cuda(), device_ids=[self.device])
        self.denoise_fn_without_ddp = self.denoise_fn.module
        self.ema_denoise_fn = copy.deepcopy(denoise_fn).cuda()
        del denoise_fn
        self.ema_denoise_fn.eval()
        self.ema_denoise_fn.requires_grad_(False)

        shift_predictor = getattr(shift_predictor_module, self.config["shift_predictor_config"]["model"], None)(self.config["shift_predictor_config"])
        self.shift_predictor = DistributedDataParallel(copy.deepcopy(shift_predictor).cuda(), device_ids=[self.device])
        self.shift_predictor_without_ddp = self.shift_predictor.module
        self.ema_shift_predictor = copy.deepcopy(shift_predictor).cuda()
        del shift_predictor
        self.ema_shift_predictor.eval()
        self.ema_shift_predictor.requires_grad_(False)

        self.enable_amp = self.config["optimizer_config"]["enable_amp"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

        if self.config["runner_config"]["compile"]:
            self.denoise_fn = torch.compile(self.denoise_fn)

    def _build_optimizer(self):
        optimizer_config = self.config["optimizer_config"]

        self.optimizer = Adam(
            [
                {"params": self.denoise_fn_without_ddp.parameters()},
                {"params": self.shift_predictor_without_ddp.parameters()},
            ],
            lr = float(optimizer_config["lr"]),
            betas = eval(optimizer_config["adam_betas"]),
            eps = float(optimizer_config["adam_eps"]),
            weight_decay= float(optimizer_config["weight_decay"]),
        )

    def train(self):
        acc_prediction_loss = 0
        acc_final_loss = 0
        time_meter = defaultdict(float)

        display_steps = self.config["runner_config"]["display_steps"]
        while True:
            start_time_top = time.time_ns()

            self.denoise_fn.train()
            self.shift_predictor.train()
            self.optimizer.zero_grad()

            # to solve small batch size for large data
            num_iterations = self.config["runner_config"]["num_iterations"]

            for _ in range(num_iterations):

                start_time = time.time_ns()
                batch = next(self.train_dataloader_infinite_cycle)
                time_meter['load data'] += (time.time_ns() - start_time) / 1e9

                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    start_time = time.time_ns()
                    output = self.gaussian_diffusion.shift_train_one_batch(
                        denoise_fn=self.denoise_fn,
                        shift_predictor=self.shift_predictor,
                        x_0=move_to_cuda(batch["x_0"]),
                        condition=move_to_cuda(batch["condition"]),
                    )
                    time_meter['forward'] += (time.time_ns() - start_time) / 1e9

                    prediction_loss = output['prediction_loss'] / num_iterations
                    final_loss = prediction_loss

                    acc_prediction_loss += prediction_loss.item()
                    acc_final_loss += final_loss.item()

                start_time = time.time_ns()
                self.scaler.scale(final_loss).backward()
                time_meter['backward'] += (time.time_ns() - start_time) / 1e9

            start_time = time.time_ns()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            time_meter['param update'] += (time.time_ns() - start_time) / 1e9

            self.step += 1

            if self.step % self.config["runner_config"]["ema_every"] == 0:
                start_time = time.time_ns()
                self.accumulate(self.config["runner_config"]["ema_decay"])
                time_meter['accumulate'] += (time.time_ns() - start_time) / 1e9

            time_meter['step'] += (time.time_ns() - start_time_top) / 1e9

            if self.step % display_steps == 0:
                info = 'rank{}: step = {}, pred = {:.5f}, final = {:.5f}'.format(
                    self.global_rank, self.step,
                    acc_prediction_loss / display_steps,
                    acc_final_loss / display_steps
                )
                print('{} '.format(info), end=' - ')
                for k, v in time_meter.items():
                    print('{}: {:.2f} secs'.format(k, v), end=', ')
                print()

                data = {'acc_prediction_loss': acc_prediction_loss, 'acc_final_loss': acc_final_loss}
                gather_data = self.gather_data(data)
                if self.global_rank == 0:
                    self.writer.add_scalar("prediction_loss", float(np.mean([data["acc_prediction_loss"] for data in gather_data])) / display_steps, self.step)
                    self.writer.add_scalar("final_loss", float(np.mean([data["acc_final_loss"] for data in gather_data])) / display_steps, self.step)
                    self.writer.add_text("log", info, self.step)

                acc_prediction_loss = 0
                acc_final_loss = 0
                time_meter.clear()

            if self.global_rank == 0 and self.step % self.config["runner_config"]["save_latest_every_steps"] == 0:
                self.save(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'latest.pt'))
            if self.global_rank == 0 and self.step % self.config["runner_config"]["save_checkpoint_every_steps"] == 0:
                self.save(os.path.join(os.path.join(self.run_path, 'checkpoints'), 'save-{}k.pt'.format(self.step // 1000)))
            if self.step % self.config["runner_config"]["evaluate_every_steps"] == 0:
                self.eval()

    def eval(self):
        torch.distributed.barrier()
        with torch.inference_mode():
            self.eval_sampler.set_epoch(self.step)
            for batch in self.eval_dataloader:
                images = self.gaussian_diffusion.shift_sample(
                    denoise_fn=self.ema_denoise_fn,
                    shift_predictor=self.ema_shift_predictor,
                    x_T=move_to_cuda(torch.randn_like(batch["x_0"])),
                    condition=move_to_cuda(batch["condition"]),
                )
                images = images.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
                images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                data = {'images': images.tolist()}
                if 'captions' in batch:
                    data.update({'captions': batch['captions']})
                if 'gts' in batch:
                    data.update({'gts': batch['gts'].tolist()})
                if 'masked_gts' in batch:
                    data.update({'masked_gts': batch['masked_gts'].tolist()})
                gather_data = self.gather_data(data)
                break

            if self.global_rank == 0:
                captions = []
                images = []
                gts = []
                masked_gts = []

                for data in gather_data:
                    images.extend(data["images"])
                    if 'captions' in data:
                        captions.extend(data['captions'])
                    if 'gts' in data:
                        gts.extend(data['gts'])
                    if 'masked_gts' in data:
                        masked_gts.extend(data['masked_gts'])

                images = np.asarray(images, dtype=np.uint8)

                captions = np.asarray(captions) if len(captions) > 0 else None
                gts = np.asarray(gts, dtype=np.uint8) if len(gts) > 0 else None
                masked_gts = np.asarray(masked_gts, dtype=np.uint8) if len(masked_gts) > 0 else None

                figure = save_image(
                    images,
                    os.path.join(self.run_path, 'samples', "sample{}k.png".format(self.step // 1000)),
                    captions=captions,
                    gts=gts,
                    masked_gts=masked_gts
                )
                self.writer.add_figure("result", figure, self.step)

    def accumulate(self, decay):
        self.denoise_fn.eval()
        ema_denoise_fn_parameter = dict(self.ema_denoise_fn.named_parameters())
        denoise_fn_parameter = dict(self.denoise_fn_without_ddp.named_parameters())

        for k in ema_denoise_fn_parameter.keys():
            if denoise_fn_parameter[k].requires_grad:
                ema_denoise_fn_parameter[k].data.mul_(decay).add_(denoise_fn_parameter[k].data, alpha=1.0 - decay)


        self.shift_predictor.eval()
        ema_shift_predictor_parameter = dict(self.ema_shift_predictor.named_parameters())
        shift_predictor_parameter = dict(self.shift_predictor_without_ddp.named_parameters())

        for k in ema_shift_predictor_parameter.keys():
            if shift_predictor_parameter[k].requires_grad:
                ema_shift_predictor_parameter[k].data.mul_(decay).add_(shift_predictor_parameter[k].data, alpha=1.0 - decay)

    def save(self, path):
        data = {
            'step': self.step,
            'denoise_fn': self.denoise_fn_without_ddp.state_dict(),
            'ema_denoise_fn': self.ema_denoise_fn.state_dict(),
            'shift_predictor': self.shift_predictor_without_ddp.state_dict(),
            'ema_shift_predictor': self.ema_shift_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, path)

        print('rank{}: step, model, optimizer and scaler saved to {}(step {}k).'.format(self.global_rank, path, self.step // 1000))

    def load(self, path):
        data = torch.load(path, map_location=torch.device('cpu'))

        self.step = data['step']
        self.denoise_fn_without_ddp.load_state_dict(data['denoise_fn'])
        self.ema_denoise_fn.load_state_dict(data['ema_denoise_fn'])
        self.shift_predictor_without_ddp.load_state_dict(data['shift_predictor'])
        self.ema_shift_predictor.load_state_dict(data['ema_shift_predictor'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.scaler.load_state_dict(data['scaler'])

        print('rank{}: step, model, optimizer and scaler restored from {}(step {}k).'.format(self.global_rank, path, self.step // 1000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--run_path', type=str, required=True)
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')

    args = parser.parse_args()
    runner = ShiftDiffusionTrainer(args)
    runner.train()
