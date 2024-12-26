import pytorch_lightning as pl
from utils import get_model, get_experiment_name, get_criterion
import torch
import warmup_scheduler


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        constraint=self.hparams.constraint
        self.model = get_model(hparams)
        self.criterion = get_criterion(self.hparams)
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        constraint=self.hparams.constraint
        optim_method=self.hparams.optim_method
        if constraint !=None:
            orth_param_list=[]
            non_orth_param_list=[]
            for name, param in self.named_parameters():
                if 'q.weight' in name or 'k.weight' in name:
                    orth_param_list.append(param)
                else:
                    non_orth_param_list.append(param)
            if 'SGD' in optim_method:
                op1=torch.optim.SGD(non_orth_param_list, lr=self.hparams.lr,  momentum=self.hparams.beta1, weight_decay=self.hparams.weight_decay)
                if optim_method=='LieGRoupSGD_HB':
                    from MomentumOptimizer_LieGroup_SOn import MomentumOptimizer_LieGroup_SOn
                    op2=MomentumOptimizer_LieGroup_SOn(orth_param_list, lr=self.hparams.lr,  momentum=self.hparams.beta1, dampening=0)
                elif optim_method=='LieGRoupSGD_NAG_SC':
                    from MomentumOptimizer_LieGroup_SOn import MomentumOptimizer_LieGroup_SOn
                    op2=MomentumOptimizer_LieGroup_SOn(orth_param_list, lr=self.hparams.lr,  momentum=self.hparams.beta1, dampening=0, NAG_SC=True)
            else:
                raise NotImplementedError()
            from MomentumOptimizer_LieGroup_SOn import CombinedOptimizer
            self.optimizer=CombinedOptimizer(op1, op2)
        else:
            if optim_method=='Adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
            elif optim_method=='AdamW':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=4*self.hparams.weight_decay/self.hparams.lr)
            elif optim_method=='SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=self.hparams.beta1, weight_decay=self.hparams.weight_decay)
            else:
                raise NotImplementedError()
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss, on_epoch=True, on_step=False)
        self.log("acc", acc, on_epoch=True, on_step=False)
        
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_acc", acc, on_epoch=True, on_step=False)
        return loss

