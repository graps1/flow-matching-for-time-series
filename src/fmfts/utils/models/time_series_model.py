import torch
from torch.utils.data import DataLoader

class TimeSeriesModel(torch.nn.Module):

    def additional_info(self):
        return {}
    
    def __repr__(self):
        info = self.additional_info()
        info_str = ", ".join(f"{key}={value}" for key, value in info.items())
        return f"{self.__class__.__name__}({info_str})"
    
    @property
    def filename(self):
        info = self.additional_info()
        info_str = "__".join(f"{key}_{value}" for key, value in info.items())
        if info_str == "": return f"{self.__class__.__name__}.pt"
        else:              return f"{self.__class__.__name__}__{info_str}.pt"
    
    def init_optimizers(self, lr):
        return { "self": torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0) }
    
    def update_optimizers(self, optimizers, lr):
        for g in optimizers["self"].param_groups:
            g["lr"] = lr
            g["initial_lr"] = lr

    def compute_loss(self, y1, x1, **kwargs):
        raise NotImplementedError()
    
    def forward(self, x, y, tx):
        raise NotImplementedError()

    def sample(self, y1, x0=None, **kwargs):
        raise NotImplementedError()

    def train_model(self, dataset_train, dataset_test, optimizers, batch_size=8, **kwargs):
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device=torch.get_default_device()))

        dataloader_test = DataLoader(
            dataset_test, 
            batch_size=1, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device=torch.get_default_device()))

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
            yield ret
