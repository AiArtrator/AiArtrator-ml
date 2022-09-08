import torch

use_gpu = False

# normally you would make your own model. Here we demonstrate with a pretrained model

model = torch.hub.load(
    "facebookresearch/pytorch_GAN_zoo:hub", "DCGAN", pretrained=True, useGPU=use_gpu
)

input_sample = torch.randn(1, 120)


# Testing
img = model.test(input_sample)
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image

print(img.shape, img.std())
print((img[img < 0]).std(), (img[img > 0]).std())
print(img)
print(img.histogram())
imgs = torchvision.utils.make_grid(img).permute(1, 2, 0).cpu().numpy()
save_image(img, "./example_onnx_maker.png")
# model.netG.eval()
# print(model.netG)

torch.onnx.export(
    model.netG,
    input_sample,
    "./tmp/model1.onnx",
    verbose=True,
    export_params=True,
    opset_version=12,
    input_names=["x"],
)
