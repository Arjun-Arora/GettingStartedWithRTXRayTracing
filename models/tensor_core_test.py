import time
import torch
from supersample_model import ESPCN
from denoise_model import KPCN_light, ApplyKernel
from tqdm import tqdm

# from torch2trt import torch2trt

input_tensor_shape = (1, 14, 1016, 1920)
output_tensor_shape = (1, 3, 1016 , 1920)

x = torch.randn(*input_tensor_shape).cuda().half()
y = torch.randn(*output_tensor_shape).cuda().half()

# model = torch.nn.Linear(D_in, D_out).cuda().half()
# model = ESPCN(1, 14, 3).cuda().half()
model = KPCN_light(input_channels=14, kernel_size=3).cuda().half()
apply_kernel = ApplyKernel(kernel_size=3).cuda().half()

opt = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in tqdm(range(50)):
    kernel = model(x)
    y_hat = apply_kernel.forward(x[:, :3], kernel, padding=True)
    
    loss = torch.nn.functional.mse_loss(y_hat, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

# with torch.autograd.profiler.profile(use_cuda=True) as prof:

model.eval()
torch.cuda.current_stream().synchronize()

# kernel = torch.randn((1, 9, 1016, 1920)).cuda()
t0 = time.time()
for i in range(50):
    kernel = model(x)
    y_hat = apply_kernel.forward(x, kernel, padding=True)

torch.cuda.synchronize()
t1 = time.time()

ms_trt = 1000.0 * (t1 - t0) / 50.0
print(ms_trt, "ms.")

# print(prof)
