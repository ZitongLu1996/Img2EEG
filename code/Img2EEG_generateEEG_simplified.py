#!/usr/bin/env python
# coding: utf-8

"""
Generate EEG epochs from images using a trained Img2EEG model.

Users provide:
    1. an image or a list of images;
    2. an object label or one label per image;
    3. an image description or one description per image.

The object labels and descriptions can be obtained using any method. Img2EEG
only requires their text values as input.

Output shape:
    single image:   (17, 50)
    multiple images: (n_images, 17, 50)
"""

import argparse
import math
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Union

import clip
import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch import nn
from torchvision import transforms


ImageInput = Union[str, Path, Image.Image]
LabelInput = Union[str, Sequence[str]]
SentenceInput = Union[str, Sequence[str]]


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
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
            setattr(
                self,
                f"norm1_{t}",
                nn.BatchNorm2d(out_channels * self.scale),
            )
            setattr(
                self,
                f"norm2_{t}",
                nn.BatchNorm2d(out_channels * self.scale),
            )
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
                                    nn.MaxPool2d(
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                    ),
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

    # Preserve the initialization used during original model training.
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
        n = len(imgs)

        # Support checkpoints/models created with or without DataParallel.
        cornet = self.cornet.module if hasattr(self.cornet, "module") else self.cornet

        v1_outputs = cornet.V1(imgs)
        v2_outputs = cornet.V2(v1_outputs)
        v4_outputs = cornet.V4(v2_outputs)

        vis_features = torch.cat(
            (
                v1_outputs.view(n, -1),
                v2_outputs.view(n, -1),
                v4_outputs.view(n, -1),
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
        clip_features = self.clip_model.encode_image(imgs).float()
        sen_features = torch.cat(
            (clip_features, w2v_features, sen_features),
            dim=1,
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
        self.semencoder = SemEncoder(
            clip_model, n_clip, n_w2v, n_sen, n_sem
        )
        self.fc1 = nn.Linear(n_vis + n_sem, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)
        self.activation = nn.ReLU()

    def forward(
        self,
        imgs_cornet,
        imgs_clip,
        w2v_features,
        sen_features,
    ):
        vis_features = self.visencoder(imgs_cornet)
        sem_features = self.semencoder(
            imgs_clip, w2v_features, sen_features
        )
        features = torch.cat((vis_features, sem_features), dim=1)
        features = self.activation(self.fc1(features))
        features = self.activation(self.fc2(features))
        features = self.fc3(features)

        return features


transform_cornet = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

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


def _get_device(device):
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _as_list(value):
    if isinstance(value, (str, Path, Image.Image)):
        return [value]
    return list(value)


def _open_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _prepare_text_inputs(values, n_images, name):
    if isinstance(values, str):
        values = [values]

    values = list(values)

    if len(values) != n_images:
        raise ValueError(
            f"The number of {name} ({len(values)}) must equal "
            f"the number of images ({n_images})."
        )

    return values


def _get_word_feature(label, glove):
    """
    Convert one object label to a 300-dimensional GloVe feature.

    A label may contain multiple tokens, such as "golden retriever". Available
    token vectors are averaged. If no token is available, the vector for "unk"
    is used when present; otherwise a zero vector is returned.
    """
    tokens = (
        label.lower()
        .replace(",", " ")
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )

    features = [glove[token] for token in tokens if token in glove.key_to_index]

    if features:
        return np.mean(features, axis=0)

    if "unk" in glove.key_to_index:
        return glove["unk"]

    return np.zeros(glove.vector_size, dtype=np.float32)


def load_img2eeg(
    weights_path,
    glove_path,
    device="auto",
    sentence_model_name="sentence-transformers/all-mpnet-base-v2",
):
    """
    Load one trained Img2EEG model and its required text encoders.

    Parameters
    ----------
    weights_path : str or Path
        Path to a trained Img2EEG checkpoint, such as best_model_params.pt.
    glove_path : str or Path
        Word2Vec-format file containing the 300-dimensional GloVe vectors used
        during training.
    device : {"auto", "cuda", "mps", "cpu"}
        Computation device.
    sentence_model_name : str
        SentenceTransformer model used during training.

    Returns
    -------
    commonencoder, glove, mpnet, device
    """
    device = _get_device(device)

    cornet = CORnet_S().to(device)
    url = "https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth"
    ckpt_data = torch.utils.model_zoo.load_url(url)

    # The published CORnet checkpoint contains DataParallel-style keys.
    cornet = torch.nn.DataParallel(cornet)
    cornet.load_state_dict(ckpt_data["state_dict"])

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()

    commonencoder = CommonEncoder(
        cornet,
        512,
        clip_model,
        512,
        300,
        768,
        512,
        512,
        850,
    ).to(device)

    weights = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,
    )
    commonencoder.load_state_dict(weights)
    commonencoder.eval()

    glove = KeyedVectors.load_word2vec_format(
        str(glove_path),
        binary=False,
    )
    mpnet = SentenceTransformer(sentence_model_name, device=str(device))

    return commonencoder, glove, mpnet, device


def generate_eeg_via_img2eeg(
    commonencoder,
    glove,
    mpnet,
    images,
    labels,
    sentences,
    device="auto",
    batchsize=32,
):
    """
    Generate EEG epochs from one image or a batch of images.

    Parameters
    ----------
    commonencoder : CommonEncoder
        Loaded trained Img2EEG model.
    glove : gensim.models.KeyedVectors
        GloVe word vectors used during training.
    mpnet : SentenceTransformer
        Sentence encoder used during training.
    images : image path, PIL.Image, or sequence
        One image or multiple images.
    labels : str or sequence of str
        One object label per image.
    sentences : str or sequence of str
        One image description per image.
    device : {"auto", "cuda", "mps", "cpu"} or torch.device
        Computation device.
    batchsize : int
        Number of images processed at once.

    Returns
    -------
    numpy.ndarray
        Shape (17, 50) for one image or (n_images, 17, 50) for multiple images.
    """
    if isinstance(device, str):
        device = _get_device(device)

    images = _as_list(images)
    labels = _prepare_text_inputs(labels, len(images), "labels")
    sentences = _prepare_text_inputs(
        sentences,
        len(images),
        "sentences",
    )

    all_pred_eeg = []

    commonencoder.eval()

    for start in range(0, len(images), batchsize):
        end = start + batchsize

        pil_images = [_open_image(image) for image in images[start:end]]

        imgs_cornet = torch.stack(
            [transform_cornet(image) for image in pil_images]
        ).to(device)

        imgs_clip = torch.stack(
            [transform_clip(image) for image in pil_images]
        ).to(device)

        w2v_features = np.stack(
            [_get_word_feature(label, glove) for label in labels[start:end]]
        )
        w2v_features = torch.as_tensor(
            w2v_features,
            dtype=torch.float32,
            device=device,
        )

        sen_features = mpnet.encode(
            sentences[start:end],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        sen_features = torch.as_tensor(
            sen_features,
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            pred_eeg = commonencoder(
                imgs_cornet,
                imgs_clip,
                w2v_features,
                sen_features,
            )

        all_pred_eeg.append(pred_eeg.cpu().numpy())

    pred_eeg = np.concatenate(all_pred_eeg, axis=0)
    pred_eeg = pred_eeg.reshape(len(images), 17, 50)

    if len(images) == 1:
        return pred_eeg[0]

    return pred_eeg


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an EEG epoch from one image, object label, "
            "and image description."
        )
    )
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--glove_path", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--output", default="generated_eeg.npy")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    args = parser.parse_args()

    commonencoder, glove, mpnet, device = load_img2eeg(
        weights_path=args.weights_path,
        glove_path=args.glove_path,
        device=args.device,
    )

    eeg = generate_eeg_via_img2eeg(
        commonencoder=commonencoder,
        glove=glove,
        mpnet=mpnet,
        images=args.image,
        labels=args.label,
        sentences=args.sentence,
        device=device,
    )

    np.save(args.output, eeg)
    print(f"Generated EEG shape: {eeg.shape}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
