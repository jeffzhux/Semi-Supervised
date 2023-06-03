import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models import WideResNet
from datasets.build import get_cifar100
from utils.config import Config
import onnxruntime
from PIL import Image

inputs = np.array(Image.open('./data/leaf_320/tDC08/120_1363.jpg'))
inputs = np.transpose(inputs, (2, 0, 1))
inputs = np.expand_dims(inputs, axis=0)

inputs = inputs / 255.0
inputs = (inputs - 0.5) / 0.5
inputs = inputs.astype(np.float32)


model = WideResNet(num_classes = 17, depth = 28, widen_factor = 2, drop_rate = 0.0)
ckpt = torch.load('./weights/plant_disease_SL/20230509_125312_1/epoch_150.pth', map_location='cuda')
model.load_state_dict(ckpt['model_state'])
# model.eval()
{'K0': 0, 'Mg0': 1, 'OT01': 2, 'gDE03': 3, 'gDP04': 4, 'gDP06': 5, 'tDA12': 6, 'tDC01': 7, 'tDC08': 8, 'tDE03': 9, 'tDP06': 10, 'tDS07': 11, 'tID03': 12, 'tIH04': 13, 'tIH05': 14, 'tIL11': 15, 'tIL13': 16}

torch_inputs = torch.from_numpy(inputs)
output = model(torch_inputs)
print(torch_inputs[0,0,150,150])
print(torch_inputs.shape)
print(output)
print(torch.max(output, dim=-1))

print('------------------')
print(inputs[0,0,150,150])
print(inputs.shape)
onnx_model = onnxruntime.InferenceSession('./sl.onnx')
onnx_outputs = onnx_model.run(None, {"input":torch_inputs.numpy()})
print(onnx_outputs)
print(np.argmax(onnx_outputs, axis=-1))
# unique, indices, count = torch.unique(b, return_inverse =True, return_counts=True)

# if unique[0] == 0:
#     unique = unique[1:]
#     count = count[1:]
# print(count)
# print(a[unique])
# print(torch.round(count * a[unique]))
# get_num = torch.round(count * a[unique]).int()
# mask2 =torch.zeros_like(b)
# for i, num in zip(unique, get_num):
#     class_mask = torch.eq(b, i).float() # get mask by class index
#     class_rebalancing_indices = torch.topk(class_mask, num.item())[1] # class_rebalancing sampling
#     mask2[class_rebalancing_indices] = 1
    
#     print(mask2)
#     # mask2 = torch.logical_and(mask2, sub_mask).float()
#     break
#     # print((b==i).nonzero(as_tuple=True))
#     # mask2 += (b==i).nonzero(as_tuple=True)
