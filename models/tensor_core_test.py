import time
import torch
from supersample_model import ESPCN
from denoise_model import KPCN_light
from tqdm import tqdm

# from torch2trt import torch2trt

input_tensor_shape= (1, 14, 1060, 1920)
output_tensor_shape = (1,3, 1060, 1920)

x = torch.randn(*input_tensor_shape).cuda().half()
y = torch.randn(*output_tensor_shape).cuda().half()

#model = torch.nn.Linear(D_in, D_out).cuda().half()
model = ESPCN(1, 14, 3).cuda().half()
# model = KPCN_light(14).cuda().half()
opt = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in tqdm(range(50)):
    y_pred = model(x)
    #print(y_pred.size())
    loss = torch.nn.functional.mse_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

# with torch.autograd.profiler.profile(use_cuda=True) as prof:
torch.cuda.current_stream().synchronize()
t0 = time.time()
for i in range(50):
    y = model(x)
torch.cuda.synchronize()
t1 = time.time()

ms_trt = 1000.0 * (t1 - t0) / 50.0
print(ms_trt, "ms.")

# print(prof)
