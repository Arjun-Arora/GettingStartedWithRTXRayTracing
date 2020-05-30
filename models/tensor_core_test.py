import torch

N, D_in, D_out = 64, 1024, 512

x = torch.randn(N, D_in).cuda().half()
y = torch.randn(N, D_out).cuda().half()

model = torch.nn.Linear(D_in, D_out).cuda().half()

opt = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(500):
    y_pred = model(x)
    
    loss = torch.nn.functional.mse_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    y = model(x)
    torch.cuda.synchronize()

print(prof)