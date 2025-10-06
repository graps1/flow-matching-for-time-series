import torch
from torch.utils.data import DataLoader

class TimeSeriesModel(torch.nn.Module):

    def compute_loss(self, y1, x1):
        raise NotImplementedError()
    
    def forward(self, x, y, tx):
        raise NotImplementedError()
    
    def init_optimizers(self, lr):
        return { "self": torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0) }
    
    def update_optimizers(self, optimizers, lr):
        for g in optimizers["self"].param_groups:
            g["lr"] = lr
            g["initial_lr"] = lr

    def train_model(self, dataset_train, dataset_test, optimizers, batch_size=8, **kwargs):
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device='cuda'))

        dataloader_test = DataLoader(
            dataset_test, 
            batch_size=1, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device='cuda'))

        dataiter_train = iter(dataloader_train)
        dataiter_test  = iter(dataloader_test)
        opt = optimizers["self"]

        ctr = 0
        while True:
            try:    y1, x1 = next(dataiter_train)
            except: y1, x1 = next(dataiter_train := iter(dataloader_train))

            opt.zero_grad()
            loss = self.compute_loss(y1, x1, **kwargs)
            loss.backward()
            opt.step()

            ret = { "loss": { "train": loss.item() } }

            if ctr % 10 == 0:
                with torch.no_grad():
                    try:    y1, x1 = next(dataiter_test)
                    except: y1, x1 = next(dataiter_test := iter(dataloader_test))
                    loss = self.compute_loss(y1, x1, **kwargs)
                    ret["loss"] |= { "test": loss.item() }

            ctr += 1
            yield loss
