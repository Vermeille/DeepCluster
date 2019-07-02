import sys
import time
import copy

from PIL import ImageFilter

import torch
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

import numpy as np

import vqae
from entropyloss import EntropyLoss, MAEntropyLoss

from visdom import Visdom

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

class DoubleAugmentationDataset:
    def __init__(self, dataset, tfms1, tfms2):
        self.ds = dataset
        self.tfms1 = tfms1
        self.tfms2 = tfms2

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, _ = self.ds[i]
        return self.tfms1(x), self.tfms2(x)


def infinite(dlp):
    dli = iter(dlp)
    while True:
        try:
            yield next(dli)
        except Exception as e:
            print(e)
            dli = iter(dlp)

class DeepCluster:
    def __init__(self, model, dataset, K, device):
        self.K = K
        self.device = device
        self.model = model

        dl = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True,
                num_workers=32, pin_memory=True)
        self.dl = infinite(dl)

        # Just have a few batches in batchnorm before freezing it so as to have
        # meaningful running_mean and running_var
        x, _ = next(self.dl)
        for i in range(10):
            model(x.to(self.device))
            x, _ = next(self.dl)
        self.km = MiniBatchKMeans(K, verbose=True, n_init=10)
        self.pca = PCA(256, whiten=True)
        self.weight = None

    def save(self, name):
        torch.save(self.labeller.state_dict(), name)

    def start_epoch(self, n_iter, equalize_weights=False):
        acts = None
        self.labeller = copy.deepcopy(self.model)
        self.labeller.eval()
        for i in range(n_iter):
            _, x = next(self.dl)
            x = x.to(self.device, non_blocking=True)
            with torch.no_grad():
                out = self.labeller(x)[1].detach().cpu().numpy()
                acts = np.concatenate([acts, out]) if acts is not None else out
        acts = self.pca.fit_transform(acts)
        acts /= np.linalg.norm(acts, axis=1, keepdims=True)
        self.km.fit(acts)

        if equalize_weights:
            self._equalize_weights()

        self.model.to_prob.reset_parameters()

    def _equalize_weights(self):
        labels = torch.FloatTensor(self.km.labels_).to(self.device)
        weight = torch.zeros(self.K).to(self.device)
        uniq, cnt = torch.unique(labels, return_counts=True)
        weight[uniq.long()] = cnt.float()
        self.weight = 1 / (weight + 1)

    def batch(self):
        x_cpu, x_no_aug_cpu = next(self.dl)
        x = x_cpu.to(device, non_blocking=True)
        x_no_aug = x_no_aug_cpu.to(device, non_blocking=True)

        with torch.no_grad():
            acts = self.labeller(x_no_aug)[1].cpu().detach().numpy()
        acts = self.pca.transform(acts)
        acts /= np.linalg.norm(acts, axis=1, keepdims=True)
        y = torch.from_numpy(self.km.predict(acts))
        y = y.long().to(self.device, non_blocking=True)

        pred = self.model(x)[0]

        loss = F.cross_entropy(pred, y, weight=self.weight)
        return loss, x_cpu, pred, y


viz = Visdom(env='clusters')
viz.close()

K=100
SZ=64
tfms1 = TF.Compose([
        TF.Resize(SZ),
        TF.RandomAffine(5, scale=(0.8, 1.2)),
        TF.RandomCrop(SZ),
        TF.RandomHorizontalFlip(),
        TF.ToTensor()
    ])

tfms2 = TF.Compose([
        TF.Resize(SZ),
        TF.CenterCrop(SZ),
        TF.ToTensor()
    ])

device = 'cuda'


ds = torchvision.datasets.ImageFolder(sys.argv[1])
ds = ForgivingDataset(ds)
ds = DoubleAugmentationDataset(ds, tfms1, tfms2)
#ds = torch.utils.data.SubsetDataset(ds, range(64))
print('dataset size:', len(ds))
model = vqae.baseline_64(K).to(device)
#opt = optim.Adam(model.parameters(), lr=3e-4)
opt = optim.SGD(model.parameters(), momentum=0.95, lr=1e-3, weight_decay=1e-5)

clustering = DeepCluster(model, ds, K, device)

iters = 0
while True:
    clustering.start_epoch((len(ds) // 128) // 4, equalize_weights=True)
    clustering.save('labeller_{}.pth'.format(iters))

    for _ in range(5000):
        opt.zero_grad()
        loss, x_cpu, pred, y = clustering.batch()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()

        if iters % 20 == 0:
            pred_cls = pred.argmax(dim=1).cpu()

            viz.line(X=[iters], Y=[loss.item()], update='append',
                    win='r_loss', opts=dict(title='Loss'))

            viz.histogram(pred_cls.cpu(),
                win='per class', opts=dict(title='per class'))

            viz.histogram(y,
                win='per class truth', opts=dict(title='per class truth'))

            for k in range(K):
                k_aff = x_cpu[pred_cls == k]
                if k_aff.shape[0] != 0:
                    viz.images(k_aff.detach(), win='aff' + str(k),
                            opts=dict(title='Class ' + str(k)))

        iters += 1
