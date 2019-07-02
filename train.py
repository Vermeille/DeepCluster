import sys
import time
import copy
import itertools

from PIL import ImageFilter

import torch
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF

import vqae
from entropyloss import EntropyLoss, MAEntropyLoss

from visdom import Visdom

viz = Visdom(env='cluster')
viz.close()

K=100
SZ=64
tfms1 = TF.Compose([
        TF.Resize(SZ),
        TF.CenterCrop(SZ),
        TF.ToTensor()
    ])

def norm_0_1(x):
    x -= x.view(3, -1).min(dim=1)[0].view(3, 1, 1)
    x /= x.view(3, -1).max(dim=1)[0].view(3, 1, 1) + 1e-4
    return x.clamp(0, 1)

tfms2 = TF.Compose([
        TF.Resize(SZ),
        TF.RandomCrop(SZ),
        TF.RandomAffine(10, (0, 0.1), (0.9, 1.2), 5),
        #TF.RandomHorizontalFlip(),
        TF.ColorJitter(0.3, 0.3, 0.3, 0.3),
        TF.RandomGrayscale(0.3),
        #TF.RandomApply([TF.Lambda(lambda x: x.filter(ImageFilter.FIND_EDGES))], p=0.5),
        TF.ToTensor(),
        TF.RandomApply([TF.Lambda(norm_0_1)], p=0.5)
    ])

device = 'cuda'

class ForgivingDataset:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        try:
            return self.ds[i]
        except Exception as e:
            print(e)
            if i < len(self):
                return self[i + 1]
            else:
                return self[0]

class AugDataset:
    def __init__(self, ds, tfms1, tfms2):
        self.ds = ds
        self.tfms1 = tfms1
        self.tfms2 = tfms2

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x = self.ds[i][0]
        return tfms1(x), tfms2(x)

ds = torchvision.datasets.ImageFolder(sys.argv[1], tfms1)
ds = ForgivingDataset(ds)
#ds = AugDataset(ds, tfms1, tfms2)
#ds = torch.utils.data.SubsetDataset(ds, range(64))
print('dataset size:', len(ds))
dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True,
        num_workers=32, pin_memory=True)
model = vqae.baseline_64(K).to(device)
opt = optim.Adam(model.parameters(), lr=3e-4)
#opt = optim.SGD(model.parameters(), momentum=0.9, lr=1e-3)

eloss = MAEntropyLoss(0.8)
#eloss = EntropyLoss()

def infinite(dlp):
    dli = iter(dlp)
    while True:
        try:
            yield next(dli)
        except Exception as e:
            print(e)
            dli = iter(dlp)


dl = infinite(dl)
x, _ = next(dl)
m = x.mean(dim=(0, 2, 3)).detach().view(3, 1, 1).to(device)
v = x.std(dim=(0, 2, 3)).detach().view(3, 1, 1).to(device)
iters = 0
while True:
    x, _ = next(dl)
    def go():
        global x
        #global xt
        x = x.to(device, non_blocking=True)
        #xt = xt.to(device, non_blocking=True)
        B = x.shape[0]
        opt.zero_grad()
        #pred, acts = model(torch.cat([x, xt], dim=0) * 2 - 1)
        pred, acts = model((x - m) / v)
        #pred, pred_t = pred[:B], pred[B:]
        #acts, acts_t = acts[:B], acts[B:]
        L = 100
        loss, cert, div = eloss(pred, lam=L)
        #kl = F.kl_div(pred, F.softmax(pred_t, dim=1))
        print(acts.mean(0).mean().item(), acts.std(0).mean().item())
        #kl = F.mse_loss(acts, acts_t)
        loss /= L
        #loss += 1 * kl
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        print(norm)
        opt.step()

        if iters <= 100 or True:
            print('catch up', iters)
            model.eval()
            with torch.no_grad():
                for i in range(3):
                    x, _ = next(dl)
                    x = x.to(device, non_blocking=True)
                    eloss(model((x - m) / v)[0])
            model.train()


        if iters % 10 == 0:
            viz.line(X=[iters], Y=[loss.item()], update='append',
                    win='r_loss', opts=dict(title='Loss'))
            viz.line(X=[iters], Y=[cert.item()], update='append',
                    win='c_loss', opts=dict(title='Certitude loss'))
            viz.line(X=[iters], Y=[div.item()], update='append',
                    win='d_loss', opts=dict(title='Diversity loss'))
            #viz.line(X=[iters], Y=[kl.item()], update='append', win='k_loss', opts=dict(title='KL'))
            viz.line(X=list(range(K)), Y=F.softmax(pred, dim=1).transpose(0, 1),
                win='per class', opts=dict(title='per class'))

            pred_cls = pred.argmax(dim=1)
            for k in range(K):
                k_order = (-pred[pred_cls, k]).argsort(dim=0)
                k_aff = x[k_order[pred_cls[k_order] == k]]
                if k_aff.shape[0] != 0:
                    viz.images(k_aff.cpu().detach(), win='aff' + str(k),
                            opts=dict(title='Class ' + str(k)))
                else:
                    viz.close(win='aff'+str(k))

    go()
    iters += 1
