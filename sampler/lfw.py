# import sys
# sys.path.append("../")
#
# from PIL import Image
# import torch
#
# import dataset as dataset_module
# from model.diffusion import GaussianDiffusion
# import model.denoise_fn as denoise_fn_module
# import model.shift_predictor as shift_predictor_module
# from utils import load_yaml, move_to_cuda
#
# device = "cuda:0"
# torch.cuda.set_device(device)
#
# config = {
#     "config_path": "../trained-models/lfw_shift/config.yml",
#     "checkpoint_path": "../trained-models/lfw_shift/checkpoint.pt",
#
#     "dataset_name": "LFW",
#     "data_path": "../data/lfw",
#     "image_channel": 3,
#     "image_size": 64,
#     "train": False,
#
#     "image_index": 666,
# }
#
# config_path = config["config_path"]
# checkpoint_path = config["checkpoint_path"]
# model_config = load_yaml(config_path)
# gaussian_diffusion = GaussianDiffusion(model_config["diffusion_config"], device=device)
# denoise_fn = getattr(denoise_fn_module, model_config["denoise_fn_config"]["model"], None)(**model_config["denoise_fn_config"])
# shift_predictor = getattr(shift_predictor_module, model_config["shift_predictor_config"]["model"], None)(model_config["shift_predictor_config"])
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# denoise_fn.load_state_dict(checkpoint['ema_denoise_fn'])
# shift_predictor.load_state_dict(checkpoint['ema_shift_predictor'])
# denoise_fn = denoise_fn.cuda()
# denoise_fn.eval()
# shift_predictor = shift_predictor.cuda()
# shift_predictor.eval()
#
# dataset_name = config["dataset_name"]
# data_path = config["data_path"]
# image_size = config["image_size"]
# image_channel = config["image_channel"]
# train = config["train"]
# dataset = getattr(dataset_module, dataset_name, None)({"data_path": data_path, "image_size": image_size, "image_channel": image_channel, "train": train})
#
# image_index = config["image_index"]
#
# with torch.inference_mode():
#     data = dataset.__getitem__(image_index)
#     gt = data["gt"]
#     # x_0 = data["x_0"].unsqueeze(0)
#     x_T = move_to_cuda(data["x_T"].unsqueeze(0))
#     condition = move_to_cuda(data["condition"].unsqueeze(0))
#
#     u = shift_predictor(condition)
#     u = u.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
#     u = u.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
#
#     image = gaussian_diffusion.shift_sample(
#         denoise_fn=denoise_fn,
#         shift_predictor=shift_predictor,
#         x_T=x_T,
#         condition=condition,
#     )
#     image = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
#     image = image.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
#
#
#     merge = Image.new('RGB', (3 * image_size , image_size), color = (255, 255, 255))
#
#     merge.paste(Image.fromarray(gt), (0, 0))
#     merge.paste(Image.fromarray(u[0]), (image_size, 0))
#     merge.paste(Image.fromarray(image[0]), (2 * image_size, 0))
#
#     merge.save("./lfw_result_"+ str(image_index) + ".png")
#
# # CUDA_VISIBLE_DEVICES=0 python3 lfw.py