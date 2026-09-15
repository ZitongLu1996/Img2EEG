#!/usr/bin/env python3
# coding: utf-8
"""Train Img2EEG models.

This version keeps the original class names, function names, variable names,
model architecture, loss, optimizer, and model-selection procedure. It only
removes unused code and exposes paths/training settings through command-line
arguments so that other researchers can run the script on their own systems.
"""

import argparse
import math
import os
import random
import time
from collections import OrderedDict

import clip
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm


def set_seed(seed=0):
    """Set random seeds for more reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic settings improve reproducibility on CUDA, although exact
    # results can still depend on PyTorch, CUDA, cuDNN, and hardware versions.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Flatten(nn.Module):
    """Flatten all dimensions except the batch dimension."""

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """Identity layer used to expose intermediate CORnet-S outputs."""

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.skip = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=2, bias=False
        )
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            out_channels, out_channels * self.scale, kernel_size=1, bias=False
        )
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels * self.scale,
            out_channels * self.scale,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            out_channels * self.scale, out_channels, kernel_size=1, bias=False
        )
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()

        for t in range(self.times):
            setattr(self, f"norm1_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm2_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm3_{t}", nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f"norm1_{t}")(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f"norm2_{t}")(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f"norm3_{t}")(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(
        OrderedDict(
            [
                (
                    "V1",
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv1",
                                    nn.Conv2d(
                                        3,
                                        64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        bias=False,
                                    ),
                                ),
                                ("norm1", nn.BatchNorm2d(64)),
                                ("nonlin1", nn.ReLU(inplace=True)),
                                (
                                    "pool",
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                ),
                                (
                                    "conv2",
                                    nn.Conv2d(
                                        64,
                                        64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm2", nn.BatchNorm2d(64)),
                                ("nonlin2", nn.ReLU(inplace=True)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
                ("V2", CORblock_S(64, 128, times=2)),
                ("V4", CORblock_S(128, 256, times=4)),
                ("IT", CORblock_S(256, 512, times=2)),
                (
                    "decoder",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("avgpool", nn.AdaptiveAvgPool2d(1)),
                                ("flatten", Flatten()),
                                ("linear", nn.Linear(512, 1000)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    # Keep the original initialization exactly. Linear layers intentionally use
    # PyTorch's default initialization, matching the models already trained.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model


class VisEncoder(nn.Module):
    def __init__(self, cornet, n_output):
        super(VisEncoder, self).__init__()

        self.cornet = cornet
        for param in self.cornet.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(351232, n_output)
        self.activation = nn.ReLU()

    def forward(self, imgs):
        N = len(imgs)

        # The original code used self.cornet.module because CORnet-S was wrapped
        # in DataParallel. This fallback also allows the same script to run on a
        # CPU or a single non-DataParallel device.
        cornet = self.cornet.module if hasattr(self.cornet, "module") else self.cornet

        v1_outputs = cornet.V1(imgs)
        v2_outputs = cornet.V2(v1_outputs)
        v4_outputs = cornet.V4(v2_outputs)
        vis_features = torch.cat(
            (
                v1_outputs.view(N, -1),
                v2_outputs.view(N, -1),
                v4_outputs.view(N, -1),
            ),
            dim=1,
        )
        vis_features = self.activation(self.fc(vis_features))

        return vis_features


class SemEncoder(nn.Module):
    def __init__(self, clip_model, n_clip, n_w2v, n_sen, n_output):
        super(SemEncoder, self).__init__()

        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(n_clip + n_w2v + n_sen, n_output)
        self.activation = nn.ReLU()

    def forward(self, imgs, w2v_features, sen_features):
        clip_features = self.clip_model.encode_image(imgs)
        sen_features = torch.cat(
            (clip_features, w2v_features, sen_features), dim=1
        )
        sem_features = self.activation(self.fc(sen_features))

        return sem_features


class CommonEncoder(nn.Module):
    def __init__(
        self,
        cornet,
        n_vis,
        clip_model,
        n_clip,
        n_w2v,
        n_sen,
        n_sem,
        n_hidden,
        n_output,
    ):
        super(CommonEncoder, self).__init__()

        self.visencoder = VisEncoder(cornet, n_vis)
        self.semencoder = SemEncoder(clip_model, n_clip, n_w2v, n_sen, n_sem)
        self.fc1 = nn.Linear(n_vis + n_sem, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)
        self.activation = nn.ReLU()

    def forward(self, imgs_cornet, imgs_clip, w2v_features, sen_features):
        vis_features = self.visencoder(imgs_cornet)
        sem_features = self.semencoder(imgs_clip, w2v_features, sen_features)
        features = torch.cat((vis_features, sem_features), dim=1)
        features = self.activation(self.fc1(features))
        features = self.activation(self.fc2(features))
        features = self.fc3(features)

        return features


class Data4Model(torch.utils.data.Dataset):
    def __init__(
        self,
        state="training",
        sub_index=1,
        transform_cornet=None,
        transform_clip=None,
        data_path=".",
    ):
        super(Data4Model, self).__init__()

        imgs = np.load(
            os.path.join(data_path, "GetData", state + "_imgpaths.npy")
        ).tolist()
        w2v_features = np.load(
            os.path.join(data_path, "GetData", state + "_word_features.npy")
        ).tolist()
        sen_features = np.load(
            os.path.join(data_path, "GetData", state + "_sen_features.npy")
        ).tolist()

        mean = np.load(
            os.path.join(data_path, "GetData", "preprocessed_mean_overall.npy")
        )
        std = np.load(
            os.path.join(data_path, "GetData", "preprocessed_std_overall.npy")
        )
        eeg = np.load(
            os.path.join(
                data_path,
                "preprocessed_eeg_data",
                "sub-" + str(sub_index).zfill(2) + "_" + state + ".npy",
            )
        )
        eeg = (eeg - mean[sub_index - 1]) / std[sub_index - 1]

        if not (len(imgs) == len(w2v_features) == len(sen_features) == len(eeg)):
            raise ValueError(
                f"Different sample counts for subject {sub_index}, state={state}: "
                f"images={len(imgs)}, word={len(w2v_features)}, "
                f"sentence={len(sen_features)}, EEG={len(eeg)}"
            )

        self.imgs = imgs
        self.w2v_features = w2v_features
        self.sen_features = sen_features
        self.eeg = eeg
        self.transform_cornet = transform_cornet
        self.transform_clip = transform_clip

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        # Load the image once and apply both transforms to the same RGB image.
        with Image.open(self.imgs[item]) as img:
            img = img.convert("RGB")
            imgs_cornet = self.transform_cornet(img)
            imgs_clip = self.transform_clip(img)

        w2v_features = torch.tensor(self.w2v_features[item]).float()
        sen_features = torch.tensor(self.sen_features[item]).float()
        eeg = torch.tensor(self.eeg[item]).float()

        return imgs_cornet, imgs_clip, w2v_features, sen_features, eeg


criterion = nn.MSELoss()


def get_loss(pred, eeg, criterion):
    loss = criterion(pred, eeg)
    return loss


def train_and_test(
    commonencoder,
    weightspath,
    criterion,
    optimizer,
    transform_cornet,
    transform_clip,
    sub_index=1,
    batchsize=64,
    num_epochs=100,
    data_path=".",
    num_workers=0,
    device="cuda",
):
    os.makedirs(weightspath, exist_ok=True)

    train_dataset = Data4Model(
        state="training",
        sub_index=sub_index,
        transform_cornet=transform_cornet,
        transform_clip=transform_clip,
        data_path=data_path,
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataset = Data4Model(
        state="test",
        sub_index=sub_index,
        transform_cornet=transform_cornet,
        transform_clip=transform_clip,
        data_path=data_path,
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    since = time.time()

    loss_save = np.zeros([num_epochs])
    corr_save = np.zeros([num_epochs])

    best_model_params_path = os.path.join(weightspath, "best_model_params.pt")
    best_corr = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        commonencoder.train()
        running_loss = 0.0
        niterates = 0

        for imgs_cornet, imgs_clip, w2v_features, sen_features, eeg in tqdm(
            train_data_loader
        ):
            imgs_cornet = imgs_cornet.to(device)
            imgs_clip = imgs_clip.to(device)
            w2v_features = w2v_features.to(device)
            sen_features = sen_features.to(device)
            eeg = eeg.to(device)

            optimizer.zero_grad()
            pred = commonencoder(
                imgs_cornet, imgs_clip, w2v_features, sen_features
            )
            loss = get_loss(pred, eeg, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            niterates += 1

        loss_save[epoch] = running_loss / niterates
        print(f"Train Loss: {running_loss / niterates:.4f}")

        commonencoder.eval()
        allgenerated_eeg = np.zeros([len(test_dataset), 850])
        alleeg = np.zeros([len(test_dataset), 850])
        index = 0

        with torch.no_grad():
            for imgs_cornet, imgs_clip, w2v_features, sen_features, eeg in test_data_loader:
                imgs_cornet = imgs_cornet.to(device)
                imgs_clip = imgs_clip.to(device)
                w2v_features = w2v_features.to(device)
                sen_features = sen_features.to(device)
                eeg = eeg.to(device)

                pred = commonencoder(
                    imgs_cornet, imgs_clip, w2v_features, sen_features
                )
                allgenerated_eeg[index] = pred.detach().cpu().numpy()[0]
                alleeg[index] = eeg.detach().cpu().numpy()[0]
                index += 1

        allgenerated_eeg = np.reshape(allgenerated_eeg, [-1])
        alleeg = np.reshape(alleeg, [-1])
        corr_save[epoch] = spearmanr(alleeg, allgenerated_eeg)[0]

        print(f"Test Corr: {corr_save[epoch]:.4f}")

        if epoch == 0:
            best_corr = corr_save[epoch]
            torch.save(commonencoder.state_dict(), best_model_params_path)

        if corr_save[epoch] > best_corr:
            best_corr = corr_save[epoch]
            torch.save(commonencoder.state_dict(), best_model_params_path)

        epoch_model_params_path = os.path.join(
            weightspath, "epoch" + str(epoch) + "_model_params.pt"
        )
        torch.save(commonencoder.state_dict(), epoch_model_params_path)

    time_elapsed = time.time() - since

    np.savetxt(os.path.join(weightspath, "loss.txt"), loss_save)
    np.savetxt(os.path.join(weightspath, "corr.txt"), corr_save)

    print(
        f"Training complete in {time_elapsed // 60:.0f}m "
        f"{time_elapsed % 60:.0f}s"
    )
    print(f"Best test corr: {best_corr:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Img2EEG models.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="Directory containing GetData/ and preprocessed_eeg_data/.",
    )
    parser.add_argument(
        "--weightspath",
        type=str,
        default="./weights/fullmodel",
        help="Root directory in which subject-specific checkpoints are saved.",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Subject indices to train. Default: 1 through 10.",
    )
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")

    torch.set_default_dtype(torch.float32)

    transform_cornet = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    cornet = CORnet_S().to(device)
    if device == "cuda":
        cornet = torch.nn.DataParallel(cornet)

    url = "https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth"
    ckpt_data = torch.utils.model_zoo.load_url(url, map_location=device)

    # Published CORnet-S weights include the "module." prefix because they were
    # saved from DataParallel. Load them directly in that case; otherwise strip
    # the prefix for CPU/MPS execution.
    if hasattr(cornet, "module"):
        cornet.load_state_dict(ckpt_data["state_dict"])
    else:
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in ckpt_data["state_dict"].items()
        }
        cornet.load_state_dict(state_dict)

    transform_clip = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    clip_model, _ = clip.load("ViT-B/32", device=device)

    for i in [sub_index - 1 for sub_index in args.subjects]:
        # Reset the seed before each subject so each subject is reproducible
        # independently when trained alone or as part of the full loop.
        set_seed(args.seed + i)

        commonencoder = CommonEncoder(
            cornet, 512, clip_model, 512, 300, 768, 512, 512, 850
        ).to(device)

        optimizer = torch.optim.Adam(commonencoder.parameters(), lr=args.lr)

        train_and_test(
            commonencoder,
            os.path.join(
                args.weightspath, "sub-" + str(i + 1).zfill(2)
            ),
            criterion,
            optimizer,
            transform_cornet,
            transform_clip,
            sub_index=i + 1,
            batchsize=args.batchsize,
            num_epochs=args.num_epochs,
            data_path=args.data_path,
            num_workers=args.num_workers,
            device=device,
        )


if __name__ == "__main__":
    main()
