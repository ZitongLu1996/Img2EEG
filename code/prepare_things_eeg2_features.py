#!/usr/bin/env python
# coding: utf-8

"""
Prepare THINGS-EEG2 inputs for Img2EEG training.

This script creates the files used by the Img2EEG training code:

GetData/
    training_imgpaths.npy
    test_imgpaths.npy
    training_objnames.npy
    test_objnames.npy
    training_word_features.npy
    test_word_features.npy
    training_imgcaptions.npy
    test_imgcaptions.npy
    training_sen_features.npy
    test_sen_features.npy
    preprocessed_mean_overall.npy
    preprocessed_std_overall.npy

The original Img2EEG pipeline used:
    - GloVe 840B 300-dimensional word vectors
    - BLIP-2 OPT-2.7B image captions
    - all-mpnet-base-v2 sentence embeddings
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import torch
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def get_device(device="auto"):
    """Select CUDA when available, otherwise use CPU."""
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def natural_sort(paths):
    """Sort paths numerically where possible."""
    def key(path):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))
        ]

    return sorted(paths, key=key)


def get_concept_folder(image_root, concept_index):
    """
    Find the folder beginning with a five-digit THINGS concept index.

    Example
    -------
    00001_aardvark_...
    """
    matches = natural_sort(
        glob.glob(str(Path(image_root) / f"{concept_index:05d}*"))
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one folder beginning with "
            f"{concept_index:05d} in {image_root}, but found {len(matches)}."
        )

    return Path(matches[0])


def get_object_name(concept_folder, concept_index):
    """
    Recover the object name using the naming convention in the original code.

    The original notebook extracted the text following the five-digit concept
    index and removed the repeated suffix in the THINGS-EEG2 folder name.
    """
    folder_name = concept_folder.name

    match = re.search(rf"{concept_index:05d}_(.*)_.*", folder_name)
    if match is None:
        raise ValueError(
            f"Could not extract an object name from folder: {folder_name}"
        )

    name = match.group(1)
    length = int((len(name) - 1) * 0.5)
    name = name[:length]

    if name and name[-1].isdigit():
        name = name[:-1]

    return name.replace("_", "")


def collect_split(
    image_root,
    n_concepts,
    images_per_concept,
):
    """
    Collect image paths and object labels in THINGS-EEG2 concept order.

    The returned path and label arrays have one entry per EEG trial/image.
    """
    imgpaths = []
    names = []

    for concept_index in tqdm(
        range(1, n_concepts + 1),
        desc=f"Reading {Path(image_root).name}",
    ):
        concept_folder = get_concept_folder(image_root, concept_index)
        concept_images = natural_sort(concept_folder.glob("*.jpg"))

        if len(concept_images) < images_per_concept:
            raise ValueError(
                f"{concept_folder} contains {len(concept_images)} JPG files; "
                f"{images_per_concept} are required."
            )

        object_name = get_object_name(concept_folder, concept_index)

        selected_images = concept_images[:images_per_concept]
        imgpaths.extend(str(path.resolve()) for path in selected_images)
        names.extend([object_name] * images_per_concept)

    return np.asarray(imgpaths), np.asarray(names)


def generate_word_features(names, glove_model):
    """
    Convert object labels to 300-dimensional GloVe vectors.

    To preserve the original training procedure, the complete object label is
    looked up as a single token. Out-of-vocabulary labels use the 'unk' vector.
    """
    if "unk" not in glove_model.key_to_index:
        raise KeyError(
            "The GloVe file must contain an 'unk' vector because the original "
            "pipeline used it for out-of-vocabulary object labels."
        )

    word_features = np.zeros(
        (len(names), glove_model.vector_size),
        dtype=np.float32,
    )

    n_unknown = 0

    for index, name in enumerate(names):
        lookup_name = str(name)

        if lookup_name not in glove_model.key_to_index:
            lookup_name = "unk"
            n_unknown += 1

        word_features[index] = glove_model[lookup_name]

    print(
        f"Object labels: {len(names)}; "
        f"out-of-vocabulary labels: {n_unknown}"
    )

    return word_features


def generate_captions(
    imgpaths,
    processor,
    model,
    device,
):
    """Generate one BLIP-2 caption for each image."""
    imgcaptions = []

    for imgpath in tqdm(imgpaths, desc="Generating BLIP-2 captions"):
        with Image.open(imgpath) as image:
            image = image.convert("RGB")
            inputs = processor(
                images=image,
                return_tensors="pt",
            )

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        inputs = {
            key: value.to(device=device, dtype=dtype)
            if torch.is_floating_point(value)
            else value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = model.generate(**inputs)

        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        imgcaptions.append(generated_text)

    return np.asarray(imgcaptions)


def generate_sentence_features(
    imgcaptions,
    model,
    batchsize=64,
):
    """Encode captions with all-mpnet-base-v2."""
    sen_features = model.encode(
        imgcaptions.tolist(),
        batch_size=batchsize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return np.asarray(sen_features, dtype=np.float32)


def calculate_eeg_mean_std(
    eeg_root,
    n_subjects,
):
    """
    Calculate one global training-set mean and standard deviation per subject.

    Expected filenames:
        sub-01_training.npy
        sub-02_training.npy
        ...
    """
    mean = np.zeros(n_subjects, dtype=np.float32)
    std = np.zeros(n_subjects, dtype=np.float32)

    for sub in range(1, n_subjects + 1):
        eeg_path = Path(eeg_root) / f"sub-{sub:02d}_training.npy"

        if not eeg_path.exists():
            raise FileNotFoundError(f"EEG file not found: {eeg_path}")

        data = np.load(eeg_path, mmap_mode="r")
        mean[sub - 1] = np.mean(data)
        std[sub - 1] = np.std(data)

        if std[sub - 1] == 0:
            raise ValueError(f"EEG standard deviation is zero: {eeg_path}")

        print(
            f"sub-{sub:02d}: shape={data.shape}, "
            f"mean={mean[sub - 1]:.6f}, std={std[sub - 1]:.6f}"
        )

    return mean, std


def save_split_files(
    output_root,
    split,
    imgpaths,
    names,
    word_features,
):
    """Save image paths, object labels, and GloVe features."""
    np.save(output_root / f"{split}_imgpaths.npy", imgpaths)
    np.save(output_root / f"{split}_objnames.npy", names)
    np.save(output_root / f"{split}_word_features.npy", word_features)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare THINGS-EEG2 features for Img2EEG training."
    )

    parser.add_argument(
        "--training_image_root",
        required=True,
        help="Folder containing the 1,654 THINGS-EEG2 training concept folders.",
    )
    parser.add_argument(
        "--test_image_root",
        required=True,
        help="Folder containing the 200 THINGS-EEG2 test concept folders.",
    )
    parser.add_argument(
        "--eeg_root",
        required=True,
        help="Folder containing preprocessed sub-XX_training.npy files.",
    )
    parser.add_argument(
        "--glove_path",
        required=True,
        help="Word2Vec-format GloVe 840B 300d file.",
    )
    parser.add_argument(
        "--output_root",
        default="GetData",
        help="Output folder used by the Img2EEG training script.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--n_subjects", type=int, default=10)
    parser.add_argument("--sentence_batchsize", type=int, default=64)
    parser.add_argument(
        "--skip_captions",
        action="store_true",
        help=(
            "Do not run BLIP-2. Use this after captions have already been "
            "generated and saved."
        ),
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # 1. Collect image paths and object labels.
    training_imgpaths, training_names = collect_split(
        args.training_image_root,
        n_concepts=1654,
        images_per_concept=10,
    )
    test_imgpaths, test_names = collect_split(
        args.test_image_root,
        n_concepts=200,
        images_per_concept=1,
    )

    print(f"Training images: {len(training_imgpaths)}")
    print(f"Test images: {len(test_imgpaths)}")

    # 2. Generate GloVe object-label features.
    glove_model = KeyedVectors.load_word2vec_format(
        args.glove_path,
        binary=False,
    )

    training_word_features = generate_word_features(
        training_names,
        glove_model,
    )
    test_word_features = generate_word_features(
        test_names,
        glove_model,
    )

    save_split_files(
        output_root,
        "training",
        training_imgpaths,
        training_names,
        training_word_features,
    )
    save_split_files(
        output_root,
        "test",
        test_imgpaths,
        test_names,
        test_word_features,
    )

    # 3. Generate BLIP-2 captions.
    training_caption_path = output_root / "training_imgcaptions.npy"
    test_caption_path = output_root / "test_imgcaptions.npy"

    if args.skip_captions:
        if not training_caption_path.exists() or not test_caption_path.exists():
            raise FileNotFoundError(
                "--skip_captions was used, but caption files do not exist."
            )

        training_imgcaptions = np.load(training_caption_path)
        test_imgcaptions = np.load(test_caption_path)
    else:
        processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        caption_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.float32
            ),
        ).to(device)
        caption_model.eval()

        training_imgcaptions = generate_captions(
            training_imgpaths,
            processor,
            caption_model,
            device,
        )
        np.save(training_caption_path, training_imgcaptions)

        test_imgcaptions = generate_captions(
            test_imgpaths,
            processor,
            caption_model,
            device,
        )
        np.save(test_caption_path, test_imgcaptions)

        del caption_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Generate MPNet sentence features.
    sentence_model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device=str(device),
    )

    training_sen_features = generate_sentence_features(
        training_imgcaptions,
        sentence_model,
        batchsize=args.sentence_batchsize,
    )
    test_sen_features = generate_sentence_features(
        test_imgcaptions,
        sentence_model,
        batchsize=args.sentence_batchsize,
    )

    np.save(
        output_root / "training_sen_features.npy",
        training_sen_features,
    )
    np.save(
        output_root / "test_sen_features.npy",
        test_sen_features,
    )

    # 5. Calculate training EEG normalization values.
    mean, std = calculate_eeg_mean_std(
        args.eeg_root,
        args.n_subjects,
    )

    np.save(output_root / "preprocessed_mean_overall.npy", mean)
    np.save(output_root / "preprocessed_std_overall.npy", std)

    print("\nFinished preparing Img2EEG training inputs.")
    print(f"Files saved to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
