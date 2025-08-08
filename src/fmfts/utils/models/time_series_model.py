import torch
from torch.utils.data import DataLoader

class TimeSeriesModel(torch.nn.Module):
    
    def compute_loss(self, y1, x1):
        raise NotImplementedError()
    
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

            opt.zero_grad()
            loss = self.compute_loss(y1, x1, **kwargs)
            # loss.backward(create_graph=True)
            loss.backward()
            opt.step()

            ctr += 1
            yield loss.item()
