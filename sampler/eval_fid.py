import argparse

import torch

import dataset as dataset_module
from metric.fid.fid_metric import FIDMetric
from sampler.base_sampler import BaseSampler

class Sampler(BaseSampler):
    def __init__(self, args):
        super().__init__(args)
        print('rank{}: sampler initialized.'.format(self.global_rank))

    def _build_dataloader(self):
        dataset_config = self.config["dataset_config"]
        self.dataset = getattr(dataset_module, dataset_config["dataset_name"], None)(dataset_config)

        self.dataset_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=self.dataset,
            num_replicas=self.global_world_size,
            rank=self.global_rank,
            shuffle=False,
            drop_last=False
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            sampler=self.dataset_sampler,
            pin_memory=False,
            collate_fn=self.dataset.collate_fn,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            drop_last=False,
        )

    def _build_model(self):
        self.fid_metric = FIDMetric(
            dims=2048,
            inception_path=self.config["inception_path"],
            device=self.device,
            target_path=self.config["target_path"],
            img_save_path=None,
        )

    def start(self):
        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataloader):
                if self.global_rank == 0:
                    print(batch_id)
                x_0 = batch["x_0"]
                self.fid_metric.process(x_0.to(self.device), None, normalize_input=False)

        fid_results = self.fid_metric.all_gather_results(self.global_world_size)
        if self.global_rank == 0:
            fid = self.fid_metric.compute_metrics(fid_results)
            print(fid)
        torch.distributed.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.config = {
        "inception_path": "./misc/pt_inception-2015-12-05-6726825d.pth",
        "target_path": "./misc/cifar10_val_fid_stats.pt",

        "dataset_config": {
            "dataset_name": "CIFAR",
            "data_path": "./data/cifar",
            "image_channel": 3,
            "image_size": 32,
            "train": True,
        },

        "batch_size": 100,
        "num_workers": 2,
    }

    runner = Sampler(args)
    runner.start()
