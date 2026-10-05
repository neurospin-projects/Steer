"""
Image-dataset backends for the MultiBench λ-conditioned SSL benchmark.

The affect datasets (mosi / mosei / humor) are PRE-EXTRACTED SEQUENCE FEATURES
of shape (B, T, p) fed to a Conv1d+Transformer encoder. AV-MNIST and ENRICO are
RAW IMAGES, so they need CNN encoders instead. Everything DOWNSTREAM of the
encoder is unchanged: the CNN encoders output a normalised `ENC_OUT_DIM`-d vector
(= benchmark_multibench.ENC_DIM_XFMR = 40), which is exactly the interface the
existing projection heads / LoRA heads / SimCLR losses / linear probe / summary
already consume. Nothing about the methods changes — only the per-modality
encoder and the data loader.

Integration points (all in benchmark_multibench.py + probe.py):
    IMAGE_DATASETS               registry: modality names, #classes, layout
    make_image_encoders(...)     -> (enc0, enc1)   [used by _make_encoders]
    image_ssl_loader(...)        -> DataLoader yielding (aug1, aug2)
    image_probe_loader(...)      -> DataLoader yielding (X=[m0,m1], y)

Encoders follow MultiBench's standard architectures for comparability:
    AV-MNIST : LeNet(1,6,3) [image]  and  LeNet(1,6,5) [audio spectrogram]
    ENRICO   : VGG11Slim(pretrained, frozen features) x2
each followed by a Linear adapter to ENC_OUT_DIM so the projection-head interface
(in_dim = 40) is identical across every dataset in the benchmark.

SimCLR augmentations are per-modality-appropriate and label-preserving:
    digit / UI screenshot / wireframe : RandomResizedCrop + small rotation + noise
    audio spectrogram                 : RandomResizedCrop + noise  (no rotation/flip)
    (never horizontal-flip UI layouts; never colour-jitter grayscale/spectrograms)
"""
import csv
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Must match benchmark_multibench.ENC_DIM_XFMR. Defined here to avoid a circular
# import (benchmark_multibench imports from this module).
ENC_OUT_DIM = 40


# Encoders  (MultiBench-standard backbones + Linear adapter -> ENC_OUT_DIM)

class GlobalPooling2D(nn.Module):
    """2-D global average pooling — verbatim from MultiBench unimodals/common_models.py."""
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        return x.view(x.size(0), -1)


class LeNet(nn.Module):
    """LeNet — verbatim from MultiBench unimodals/common_models.py (avmnist backbone).

    LeNet(1, 6, 3) -> 48-d ;  LeNet(1, 6, 5) -> 192-d  (output = args_channels * 2**additional_layers).
    """
    def __init__(self, in_channels, args_channels, additional_layers, squeeze_output=True):
        super().__init__()
        convs = [nn.Conv2d(in_channels, args_channels, kernel_size=5, padding=2, bias=False)]
        bns   = [nn.BatchNorm2d(args_channels)]
        gps   = [GlobalPooling2D()]
        for i in range(additional_layers):
            convs.append(nn.Conv2d((2 ** i) * args_channels, (2 ** (i + 1)) * args_channels,
                                    kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(args_channels * (2 ** (i + 1))))
            gps.append(GlobalPooling2D())
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)
        self.gps   = nn.ModuleList(gps)
        self.sq_out = squeeze_output
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        out = x
        for i in range(len(self.convs)):
            out = F.relu(self.bns[i](self.convs[i](out)))
            out = F.max_pool2d(out, 2)
        # out: (B, C_last, H', W') — collapse spatial dims to (B, C_last)
        out = out.view(out.size(0), out.size(1), -1).mean(2)
        return out


class VGG11SlimEncoder(nn.Module):
    """MultiBench VGG11Slim, rewritten for the modern torchvision weights API.

    Pretrained (ImageNet) VGG11-bn features, optionally frozen, classifier replaced
    by a Linear to `out_dim`. Falls back to random init if weights can't be fetched
    (e.g. offline compute node) — a warning is printed so the run is still valid but
    flagged as not-pretrained.
    """
    def __init__(self, out_dim, dropoutp=0.2, freeze_features=True):
        super().__init__()
        from torchvision.models import vgg11_bn
        pretrained_ok = True
        try:
            from torchvision.models import VGG11_BN_Weights
            model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        except Exception as e:  # offline / no cached weights
            warnings.warn(f"VGG11-bn pretrained weights unavailable ({e}); using random "
                          f"init and TRAINING the features (not MultiBench-comparable — "
                          f"pre-cache the weights on a login node for the real runs).")
            model = vgg11_bn(weights=None)
            pretrained_ok = False
        # insert dropout after each ReLU in the feature extractor (matches MultiBench)
        feats = []
        for f in model.features:
            feats.append(f)
            if isinstance(f, nn.ReLU):
                feats.append(nn.Dropout(p=dropoutp))
        model.features = nn.Sequential(*feats)
        model.classifier = nn.Linear(512 * 7 * 7, out_dim)
        # Freeze features only when they are actually pretrained — freezing random
        # weights would leave the encoder useless.
        freeze = freeze_features and pretrained_ok
        for p in model.features.parameters():
            p.requires_grad = not freeze
        self.model = model

    def forward(self, x):
        return self.model(x)


class ImageModalityEncoder(nn.Module):
    """Wrap a CNN backbone with a Linear adapter -> ENC_OUT_DIM and L2-normalise.

    Presents the exact same interface as TransformerEncoder: forward(x, lam=None)
    returning a unit-norm (B, ENC_OUT_DIM) tensor. With lora=False, `lam` is
    accepted and ignored — that is the film_mode='proj'/'none' case, where lambda
    conditioning lives in the projection head.

    With lora=True the adapter becomes a LoRADualLayer and the encoder gains
    forward_mix(x, w_r, w_u), i.e. the PaLoRA form

        theta(w) = W + (alpha/r) * [ w_R * A_R B_R + w_U * A_U B_U ]

    which is what simplex / simplex_enc_decomp_R need on approach 5: the ENCODER
    itself is preference-conditioned. This mirrors LoRAAlexNetEncoder, which puts
    the same layer on AlexNet's classifier for the trifeature setting. Without it
    those film_modes die on image datasets with
    "AttributeError: 'ImageModalityEncoder' object has no attribute 'forward_mix'".

    Note the adapter is forced to exist under lora=True even when
    backbone_dim == out_dim (ENRICO's VGG), since an nn.Identity has no weights to
    adapt and lambda would silently do nothing.
    """
    def __init__(self, backbone, backbone_dim, out_dim=ENC_OUT_DIM,
                 lora: bool = False, lora_rank: int = 4, lora_alpha=None):
        super().__init__()
        self.backbone = backbone
        self.lora = lora
        if lora:
            from pareto_ssl.networks import LoRADualLayer
            self.adapter = LoRADualLayer(backbone_dim, out_dim, lora_rank, alpha=lora_alpha)
        else:
            self.adapter = (nn.Identity() if backbone_dim == out_dim
                            else nn.Linear(backbone_dim, out_dim))

    def forward(self, x, lam=None):
        h = self.backbone(x)
        if self.lora and lam is not None:
            return F.normalize(self.adapter(h, float(lam)), dim=-1)
        h = self.adapter(h) if not self.lora else self.adapter(h, 1.0)
        return F.normalize(h, dim=-1)

    def forward_mix(self, x, w_r: float, w_u: float):
        if not self.lora:
            raise RuntimeError(
                "forward_mix requires lora=True; this encoder was built lambda-blind. "
                "make_image_encoders(..., lora=True) is selected from the film_mode.")
        return F.normalize(self.adapter.forward_mix(self.backbone(x), w_r, w_u), dim=-1)

    def forward_r(self, x):
        return F.normalize(self.adapter.forward_r(self.backbone(x)), dim=-1)

    def forward_u(self, x):
        return F.normalize(self.adapter.forward_u(self.backbone(x)), dim=-1)


# Augmentations  (per-modality SimCLR views for the uniqueness terms L_Uv / L_Ut)

class _AddGaussianNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x) * self.std


def _spatial_aug(size, rotate=True, scale=(0.7, 1.0), noise_std=0.05):
    """RandomResizedCrop (+ optional small rotation) + Gaussian noise. No flip/colour."""
    tfs = [transforms.RandomResizedCrop(size, scale=scale, antialias=True)]
    if rotate:
        tfs.append(transforms.RandomRotation(15))
    tfs.append(_AddGaussianNoise(noise_std))
    return transforms.Compose(tfs)


# Dataset registry
# `enc`:   list of (kind, backbone_dim) per modality, kind in {"lenet_img","lenet_aud","vgg11"}
# `mods`:  human-readable modality names (slot0 -> enc_v, slot1 -> enc_t)
# `n_classes`: for reference / sanity (probe infers classes from labels via sklearn)
IMAGE_DATASETS = {
    "avmnist": {
        "mods": ("image", "audio"),
        "n_classes": 10,
        "enc": [("lenet_img", 48), ("lenet_aud", 192)],
    },
    "enrico": {
        "mods": ("screenshot", "wireframe"),
        "n_classes": 20,
        "enc": [("vgg11", None), ("vgg11", None)],
    },
}


def is_image_dataset(dataset: str) -> bool:
    return dataset in IMAGE_DATASETS


def make_image_encoders(dataset: str, device: str, out_dim: int = ENC_OUT_DIM,
                        freeze_vgg: bool = True, lora: bool = False,
                        lora_rank: int = 4, lora_alpha=None):
    """Build the two per-modality CNN encoders for an image dataset.

    freeze_vgg=False lets the ENRICO VGG conv features fine-tune (ImageNet features
    are out-of-domain for UI screenshots/wireframes, so unfreezing usually helps —
    at the cost of overfitting risk on the small ENRICO set).
    """
    kw = dict(lora=lora, lora_rank=lora_rank, lora_alpha=lora_alpha)
    def _build(kind, bdim):
        if kind == "lenet_img":
            return ImageModalityEncoder(LeNet(1, 6, 3), 48, out_dim, **kw)
        if kind == "lenet_aud":
            return ImageModalityEncoder(LeNet(1, 6, 5), 192, out_dim, **kw)
        if kind == "vgg11":
            # VGG11Slim outputs `out_dim` directly (classifier Linear -> out_dim)
            return ImageModalityEncoder(VGG11SlimEncoder(out_dim, freeze_features=freeze_vgg),
                                        out_dim, out_dim, **kw)
        raise ValueError(f"unknown encoder kind: {kind}")
    (k0, d0), (k1, d1) = IMAGE_DATASETS[dataset]["enc"]
    return _build(k0, d0).to(device), _build(k1, d1).to(device)


# Data loading  (AV-MNIST)
# AV-MNIST layout (MultiBench datasets/avmnist/get_data.py):
#   <root>/image/train_data.npy  (60000, 784)  -> reshape (28,28)
#   <root>/audio/train_data.npy  (60000, 112, 112)
#   <root>/train_labels.npy      (60000,)
#   <root>/image|audio/test_data.npy, <root>/test_labels.npy  (10000, ...)
# Splits match MultiBench: train = [0:55000], val = [55000:60000], test = full test.

def _load_avmnist_arrays(root, split):
    which = "test" if split == "test" else "train"
    img = np.load(os.path.join(root, "image", f"{which}_data.npy")).astype(np.float32)
    aud = np.load(os.path.join(root, "audio", f"{which}_data.npy")).astype(np.float32)
    lab = np.load(os.path.join(root, f"{which}_labels.npy")).astype(np.int64)
    img = (img / 255.0).reshape(-1, 1, 28, 28)      # (N,1,28,28), normalised
    aud = (aud / 255.0).reshape(-1, 1, 112, 112)    # (N,1,112,112), normalised
    if split == "train":
        sl = slice(0, 55000)
    elif split in ("val", "valid"):
        # probe.py asks for "valid"; matching only "val" silently fell through to
        # the whole 60k train array, 55k of which the encoder had already seen.
        sl = slice(55000, 60000)
    else:
        sl = slice(0, len(img))
    return img[sl], aud[sl], lab[sl]


class _AVMNIST(Dataset):
    """Returns (X=[image, audio], label). Optionally applies SSL augmentation twice."""
    def __init__(self, root, split, ssl=False):
        self.img, self.aud, self.lab = _load_avmnist_arrays(root, split)
        self.ssl = ssl
        self.aug_img = _spatial_aug(28,  rotate=True,  scale=(0.7, 1.0), noise_std=0.05)
        self.aug_aud = _spatial_aug(112, rotate=False, scale=(0.8, 1.0), noise_std=0.05)

    def __len__(self):
        return len(self.lab)

    def _get(self, i):
        return torch.from_numpy(self.img[i]), torch.from_numpy(self.aud[i])

    def __getitem__(self, i):
        v, t = self._get(i)
        if self.ssl:
            aug1 = [self.aug_img(v), self.aug_aud(t)]
            aug2 = [self.aug_img(v), self.aug_aud(t)]
            return aug1, aug2
        return [v, t], int(self.lab[i])


def _collate_ssl(batch):
    aug1 = [torch.stack([b[0][m] for b in batch]) for m in range(2)]
    aug2 = [torch.stack([b[1][m] for b in batch]) for m in range(2)]
    return aug1, aug2


def _collate_probe(batch):
    X = [torch.stack([b[0][m] for b in batch]) for m in range(2)]
    y = torch.tensor([b[1] for b in batch])
    return X, y


# Data loading  (ENRICO)
# ENRICO layout (MultiBench datasets/enrico/dataset):
#   <root>/design_topics.csv         (screen_id, topic)  -> 20 topic classes
#   <root>/screenshots/<id>.jpg      RGB screenshot
#   <root>/wireframes/<id>.png       RGB wireframe
# We replicate MultiBench's EnricoDataset exactly (same corrupted-file IGNORES,
# same seed-42 shuffle, same 65/15/20 split, same Resize((256,128))+ToTensor) so
# splits are comparable to published numbers. The class-balancing WeightedRandom-
# Sampler and noise-augmented test set from MultiBench's get_dataloader are NOT
# used here — SSL wants a plain shuffle and the probe wants the clean split.
_ENRICO_IGNORES = {"50105", "50109"}
_ENRICO_SPLIT = {"train": (0.0, 0.65), "val": (0.65, 0.80), "test": (0.80, 1.0)}
_ENRICO_SEED = 42
_ENRICO_H, _ENRICO_W = 256, 128


def _enrico_examples(root):
    with open(os.path.join(root, "design_topics.csv")) as f:
        rows = list(csv.DictReader(f))
    return [e for e in rows if e["screen_id"] not in _ENRICO_IGNORES]


class _Enrico(Dataset):
    """Returns (X=[screenshot, wireframe], label). Optionally augments twice for SSL."""
    def __init__(self, root, split, ssl=False):
        self.examples = _enrico_examples(root)
        self.ssl = ssl
        self.img_dir = os.path.join(root, "screenshots")
        self.wf_dir = os.path.join(root, "wireframes")
        topics = sorted({e["topic"] for e in self.examples})
        self.topic2idx = {t: i for i, t in enumerate(topics)}  # 20 classes
        keys = list(range(len(self.examples)))
        random.Random(_ENRICO_SEED).shuffle(keys)
        lo, hi = _ENRICO_SPLIT["val" if split.startswith("val") else split]
        n = len(self.examples)
        self.keys = keys[int(n * lo):int(n * hi)]
        self.base = transforms.Compose([
            transforms.Resize((_ENRICO_H, _ENRICO_W)),
            transforms.ToTensor(),
        ])
        self.aug = _spatial_aug((_ENRICO_H, _ENRICO_W), rotate=True, scale=(0.7, 1.0), noise_std=0.05)

    def __len__(self):
        return len(self.keys)

    def _load(self, i):
        e = self.examples[self.keys[i]]
        sid = e["screen_id"]
        img = self.base(Image.open(os.path.join(self.img_dir, sid + ".jpg")).convert("RGB"))
        wf = self.base(Image.open(os.path.join(self.wf_dir, sid + ".png")).convert("RGB"))
        return img, wf, self.topic2idx[e["topic"]]

    def __getitem__(self, i):
        img, wf, lab = self._load(i)
        if self.ssl:
            return [self.aug(img), self.aug(wf)], [self.aug(img), self.aug(wf)]
        return [img, wf], int(lab)


# dataset dispatch

def _build_dataset(dataset, root, split, ssl):
    if dataset == "avmnist":
        return _AVMNIST(root, split, ssl=ssl)
    if dataset == "enrico":
        return _Enrico(root, split, ssl=ssl)
    raise NotImplementedError(f"image dataset not wired: {dataset}")


def image_ssl_loader(dataset, root, batch_size, num_workers=4):
    ds = _build_dataset(dataset, root, "train", ssl=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True, collate_fn=_collate_ssl)


def image_probe_loader(dataset, root, split, batch_size, num_workers=4):
    ds = _build_dataset(dataset, root, split, ssl=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, collate_fn=_collate_probe)
