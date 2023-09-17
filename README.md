# ShiftDDPMs: Exploring Conditional Diffusion Models by Shifting Diffusion Trajectories

This repository is official PyTorch implementation of [ShiftDDPMs](https://arxiv.org/abs/2302.02373) (AAAI 2023).

```
@article{zhang2023shiftddpms,
  title={Shiftddpms: Exploring conditional diffusion models by shifting diffusion trajectories},
  author={Zhang, Zijian and Zhao, Zhou and Yu, Jun and Tian, Qi},
  journal={arXiv preprint arXiv:2302.02373},
  year={2023}
}
```



## Dataset

You can download all datasets from [Google Drive](https://drive.google.com/drive/folders/1RMptA_GCWJRxLInMCA4tN7KtIMwu_3xR?usp=sharing) .

You should put download under **ShiftDDPMs/data/** and unzip them.



## Install Requirements

```
pip install -r requirements.txt
```




## Training

To train regular DDPM, run this command:

```
bash scripts/dist_train_regular_diffusion.sh 1 0 4
```



To train ShiftDDPM, run this command:

```
bash scripts/dist_train_shift_diffusion.sh 1 0 4
```



You can change the config file and run path in the script file.



## Sampling

TODO
