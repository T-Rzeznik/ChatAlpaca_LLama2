import torch
#Test if GPU is setup properly with CUDA
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))