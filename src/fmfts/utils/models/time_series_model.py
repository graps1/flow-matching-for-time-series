import torch
from torch.utils.data import DataLoader

class TimeSeriesModel(torch.nn.Module):

    def compute_loss(self, y1, x1, ctr, steps=2):
        raise NotImplementedError()
    
    # def minibatch_gradient_step(self, y1, x1, ctr, opt, **kwargs):
    #     raise NotImplementedError()
    
    def forward(self, x, y, tx):
        raise NotImplementedError()

    def train_model(self, dataset, opt, batch_size=8, **kwargs):
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device='cuda'))

        dataiter = iter(dataloader)

        ctr = 0
        while True:
            try:    y1, x1 = next(dataiter)
            except: y1, x1 = next(dataiter := iter(dataloader))

            # loss = self.minibatch_gradient_step(y1, x1, ctr, opt, **kwargs)
            opt.zero_grad()
            loss = self.compute_loss(y1, x1, ctr, **kwargs)
            loss.backward()
            opt.step()

            ctr += 1
            yield loss
