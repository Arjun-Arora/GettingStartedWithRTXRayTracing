import torch
from supersample_model import ESPCN
from tqdm import tqdm
input_tensor_shape= (1,14, 1060, 1920)
output_tensor_shape = (1,3,1060,1920)

x = torch.randn(*input_tensor_shape).cuda().half()
y = torch.randn(*output_tensor_shape).cuda().half()

#model = torch.nn.Linear(D_in, D_out).cuda().half()
model = ESPCN(1,14,3).cuda().half()
opt = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in tqdm(range(100)):
    y_pred = model(x)
    #print(y_pred.size())
    loss = torch.nn.functional.mse_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    y = model(x)
    torch.cuda.synchronize()

print(prof)
