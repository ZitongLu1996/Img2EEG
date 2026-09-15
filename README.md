# Img2EEG

Img2EEG is a neural encoding framework that predicts image-evoked EEG responses from three complementary inputs:

1. the image itself;
2. an object label describing the main object in the image;
3. a sentence describing the image.

The model combines visual features from CORnet-S and CLIP with semantic features derived from GloVe object-label embeddings and MPNet sentence embeddings. It predicts one EEG epoch with 17 channels and 50 time points.

This repository contains three main scripts:

```text
prepare_things_eeg2_features.py
train_img2eeg.py
Img2EEG_generateEEG.py
```

They support the complete workflow from THINGS-EEG2 feature preparation to model training and EEG generation.

---

## Repository structure

```text
Img2EEG/
├── code/
│   ├──prepare_things_eeg2_features.py
│   ├──train_img2eeg.py
│   ├──Img2EEG_generateEEG.py
├── README.md
```

### Main scripts

| Script | Purpose |
|---|---|
| `prepare_things_eeg2_features.py` | Prepares image paths, object-label features, image captions, sentence features, and EEG normalization values |
| `train_img2eeg.py` | Trains one Img2EEG model for each participant |
| `Img2EEG_generateEEG.py` | Uses a trained Img2EEG model to generate EEG epochs from new images |

---

## Model inputs and output

For each image, Img2EEG receives:

```text
image + object label + image description
```

Example:

```python
image = "dog.jpg"
label = "dog"
sentence = "A dog is running through a grassy field."
```

The generated output has shape:

```text
17 EEG channels × 50 time points
```

For multiple images, the output has shape:

```text
n_images × 17 EEG channels × 50 time points
```

The model does not require a specific method for obtaining the object label or image description. Users may provide human annotations, existing metadata, or outputs from any image-classification or image-captioning model.

---

## Installation

A CUDA-capable GPU is recommended for feature preparation and model training.

Create a Python environment and install the required packages:

```bash
conda create -n img2eeg python=3.10
conda activate img2eeg
```

Install PyTorch following the instructions appropriate for your system, then install the remaining dependencies:

```bash
pip install numpy scipy pillow tqdm gensim
pip install torchvision torchmetrics
pip install transformers sentence-transformers
pip install git+https://github.com/openai/CLIP.git
```

The scripts also automatically download the pretrained CORnet-S weights when they are first run.

For exact reproduction, use the same package versions used to train the released models. A version-pinned `requirements.txt` or environment file is recommended for the final repository release.

---

# 1. Prepare THINGS-EEG2 features

The first script prepares the files required by the Img2EEG training code.

## Required inputs

Users need:

1. THINGS-EEG2 training images;
2. THINGS-EEG2 test images;
3. preprocessed EEG arrays;
4. 300-dimensional GloVe word vectors in Word2Vec text format.

Expected organization:

```text
project/
├── images/
│   ├── training_images/
│   │   ├── 00001_.../
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── test_images/
│       ├── 00001_.../
│       │   └── image.jpg
│       └── ...
├── preprocessed_eeg_data/
│   ├── sub-01_training.npy
│   ├── sub-01_test.npy
│   ├── sub-02_training.npy
│   ├── sub-02_test.npy
│   └── ...
└── gensim_glove_vectors.txt
```

The default pipeline assumes:

```text
1,654 training concepts × 10 images = 16,540 training images
200 test concepts × 1 image = 200 test images
10 participants
```

The order of the image paths must match the trial order of the corresponding EEG arrays.

## GloVe vectors

The released pipeline uses 300-dimensional GloVe 840B vectors converted to Word2Vec text format.

Example conversion:

```python
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec(
    glove_input_file="glove.840B.300d.txt",
    word2vec_output_file="gensim_glove_vectors.txt",
)
```

The converted file should contain an `unk` vector because out-of-vocabulary object labels are mapped to `unk`.

## Run feature preparation

```bash
python prepare_things_eeg2_features.py \
    --training_image_root images/training_images \
    --test_image_root images/test_images \
    --eeg_root preprocessed_eeg_data \
    --glove_path gensim_glove_vectors.txt \
    --output_root GetData \
    --device cuda
```

This script:

1. collects training and test image paths;
2. extracts THINGS object labels;
3. converts object labels into 300-dimensional GloVe vectors;
4. generates image captions using BLIP-2 OPT-2.7B;
5. converts captions into 768-dimensional MPNet embeddings;
6. calculates a global training-set EEG mean and standard deviation for each participant.

## Output files

```text
GetData/
├── training_imgpaths.npy
├── test_imgpaths.npy
├── training_objnames.npy
├── test_objnames.npy
├── training_word_features.npy
├── test_word_features.npy
├── training_imgcaptions.npy
├── test_imgcaptions.npy
├── training_sen_features.npy
├── test_sen_features.npy
├── preprocessed_mean_overall.npy
└── preprocessed_std_overall.npy
```

Expected array shapes:

```text
training_imgpaths.npy          (16540,)
test_imgpaths.npy              (200,)

training_word_features.npy     (16540, 300)
test_word_features.npy         (200, 300)

training_sen_features.npy      (16540, 768)
test_sen_features.npy          (200, 768)

preprocessed_mean_overall.npy  (10,)
preprocessed_std_overall.npy   (10,)
```

## Resume after caption generation

BLIP-2 caption generation is the slowest preparation step. Once these files exist:

```text
GetData/training_imgcaptions.npy
GetData/test_imgcaptions.npy
```

the remaining features can be regenerated without rerunning BLIP-2:

```bash
python prepare_things_eeg2_features.py \
    --training_image_root images/training_images \
    --test_image_root images/test_images \
    --eeg_root preprocessed_eeg_data \
    --glove_path gensim_glove_vectors.txt \
    --output_root GetData \
    --device cuda \
    --skip_captions
```

---

# 2. Train Img2EEG models

Img2EEG trains a separate model for each participant.

The training script expects:

```text
GetData/
preprocessed_eeg_data/
```

The EEG files should follow this naming convention:

```text
preprocessed_eeg_data/
├── sub-01_training.npy
├── sub-01_test.npy
├── sub-02_training.npy
├── sub-02_test.npy
└── ...
```

Each EEG sample must be flattened to 850 values:

```text
17 channels × 50 time points = 850 values
```

## Run model training

```bash
python train_img2eeg.py \
    --data_path . \
    --weightspath weights/fullmodel \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
    --batchsize 16 \
    --num_epochs 5 \
    --lr 0.0001 \
    --device cuda \
    --seed 0
```

The script preserves the original training procedure:

- frozen pretrained CORnet-S backbone;
- frozen pretrained CLIP image encoder;
- GloVe object-label features;
- MPNet sentence features;
- mean-squared-error loss;
- Adam optimization;
- model selection using Spearman correlation;
- participant-specific model checkpoints.

## Training outputs

For each participant:

```text
weights/fullmodel/sub-01/
├── best_model_params.pt
├── epoch0_model_params.pt
├── epoch1_model_params.pt
├── ...
├── loss.txt
└── corr.txt
```

`best_model_params.pt` contains the checkpoint with the highest evaluation-set Spearman correlation.

### Evaluation split terminology

The original training procedure evaluates the model after every epoch and selects the best checkpoint based on that correlation. Therefore, this split functions as a validation set during model selection. A fully independent test set should be retained when reporting final generalization performance.

---

# 3. Generate EEG from new images

After training, users can generate EEG epochs from new images.

The generation function requires:

```text
image + object label + image description
```

The label and description may be obtained using any method.

## Load a trained model

```python
from Img2EEG_generateEEG import (
    load_img2eeg,
    generate_eeg_via_img2eeg,
)

commonencoder, glove, mpnet, device = load_img2eeg(
    weights_path="weights/fullmodel/sub-01/best_model_params.pt",
    glove_path="gensim_glove_vectors.txt",
    device="cuda",
)
```

The model and feature encoders should be loaded once and reused for all images.

## Generate EEG for one image

```python
eeg = generate_eeg_via_img2eeg(
    commonencoder=commonencoder,
    glove=glove,
    mpnet=mpnet,
    images="images/example.jpg",
    labels="dog",
    sentences="A dog is running through a grassy field.",
    device=device,
)

print(eeg.shape)
```

Output:

```text
(17, 50)
```

## Generate EEG for multiple images

```python
images = [
    "images/dog.jpg",
    "images/car.jpg",
    "images/face.jpg",
]

labels = [
    "dog",
    "car",
    "face",
]

sentences = [
    "A dog is running through a grassy field.",
    "A red car is parked beside a road.",
    "A person is looking directly at the camera.",
]

eeg = generate_eeg_via_img2eeg(
    commonencoder=commonencoder,
    glove=glove,
    mpnet=mpnet,
    images=images,
    labels=labels,
    sentences=sentences,
    device=device,
)

print(eeg.shape)
```

Output:

```text
(3, 17, 50)
```

## Command-line generation

A single EEG epoch can also be generated from the terminal:

```bash
python Img2EEG_generateEEG.py \
    --weights_path weights/fullmodel/sub-01/best_model_params.pt \
    --glove_path gensim_glove_vectors.txt \
    --image images/example.jpg \
    --label dog \
    --sentence "A dog is running through a grassy field." \
    --output generated_eeg.npy \
    --device cuda
```

The saved NumPy array will have shape:

```text
(17, 50)
```

---

## Generating participant-specific and group-level EEG

Each checkpoint corresponds to one participant. Loading:

```text
sub-01/best_model_params.pt
```

generates the predicted response for participant 1, whereas loading:

```text
sub-02/best_model_params.pt
```

generates the predicted response for participant 2.

To obtain group-level predictions, generate EEG separately using each participant-specific model and then average the resulting arrays:

```python
group_eeg = np.mean(all_subject_eeg, axis=0)
```

The appropriate aggregation procedure depends on the intended analysis.

---

## Reproducibility

For the closest reproduction of the released Img2EEG models, users should use the same:

- THINGS-EEG2 images;
- image and EEG trial order;
- EEG preprocessing procedure;
- 300-dimensional GloVe 840B vectors;
- BLIP-2 OPT-2.7B checkpoint;
- MPNet sentence-transformer checkpoint;
- pretrained CORnet-S checkpoint;
- pretrained CLIP ViT-B/32 checkpoint;
- Python package versions;
- random seed;
- model hyperparameters.

Random seeds improve reproducibility but do not always guarantee numerically identical results across different GPUs, CUDA versions, cuDNN versions, or PyTorch versions.

Image captions generated by BLIP-2 may also differ across software or model revisions. For exact replication, the repository should provide the generated caption arrays and derived feature arrays when data-sharing and licensing conditions permit.

---

## Important notes

### Image ordering

The correspondence between image paths and EEG trials is essential. Before training, inspect:

```python
import numpy as np

imgpaths = np.load("GetData/training_imgpaths.npy")
print(imgpaths.shape)
print(imgpaths[:20])
```

Confirm that this order matches the first dimension of:

```text
sub-XX_training.npy
```

### Object labels

During original feature preparation, each complete THINGS object label was looked up as one GloVe token. Out-of-vocabulary labels were replaced with the `unk` vector.

During generation for new images, labels containing multiple words are tokenized, and available token vectors are averaged.

### Input normalization

Training EEG data are normalized separately for each participant using the mean and standard deviation calculated from that participant's training data:

```python
normalized_eeg = (eeg - participant_mean) / participant_std
```

Consequently, generated EEG values are in the normalized feature space used during model training.

### Pretrained backbones

CORnet-S and CLIP parameters are frozen during training. The original Img2EEG architecture and checkpoint format are retained to ensure compatibility with the released weights.

---

## Img2EEG weights download

Pretrained Img2EEG checkpoints are available here: [Img2EEG_weights_download_link](https://drive.google.com/drive/folders/17jCHii8t4ykwdcSeiNclHyQyVfrjD0y0?usp=sharing).

---

## ImageNet-SimEEG download

ImageNet-SimEEG (the large-scale Img2EEG-generated EEG responses corresponding to ImageNet image dataset) are available here: [ImageNet-SimEEG_download_link](https://drive.google.com/drive/folders/1Q2wRnFWrxvlmsibXsWdmhr4EJKrzw5yQ?usp=sharing).

---

## Citation

When using this code, please cite the Img2EEG paper:

```bibtex
@article{img2eeg,
  title   = {Img2EEG: A Scalable and Interpretable Encoding Framework for Simulating Human EEG Responses to Visual Inputs},
  author  = {Zitong Lu & Julie D. Golomb},
  journal = {bioRxiv},
  year    = {2026}
}
```

Please also cite the original datasets and pretrained models used in the pipeline, including THINGS-EEG2, CORnet-S, CLIP, GloVe, BLIP-2, and MPNet.

---

## License

```text
This code is released under the MIT License.
```

---

## Contact

For questions about the code or model, please open a GitHub issue or contact:

```text
Zitong Lu
zitonglu1996@gmail.com / zitonglu@mit.edu
```
