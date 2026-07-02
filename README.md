# 🎶 **From Discord to Harmony**: Decomposed Consonance-Based Training for Improved Audio Chord Estimation

This repository contains the official implementation of the paper:  

> Poltronieri, A., Serra, X. and Rocamora, M. (2025) ‘From Discord to Harmony: Decomposed Consonance-Based Training for Improved Audio Chord Estimation’, in Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR). Daejeon, South Korea: International Society for Music Information Retrieval, ISMIR 2025. Available at: https://arxiv.org/abs/2509.01588.


## 📦 Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/andreamust/consonance-ACE.git
   cd consonance-ACE
   ```

2. Create the conda environment (Python 3.11):

   ```bash
   conda create -n ace python=3.11
   conda activate ace
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 📂 Dataset Preparation

Our experiments use chord annotations and audio from:

* Training / Validation:
    * Isophonics dataset
    * McGill Billboard corpus

* Testing:
    * RWC Pop dataset
    * USPop dataset

Chord annotations are sourced from [ChoCo: the Chord Corpus](https://github.com/andreamust/ChoCo).

Prepare the dataset features by running:

```bash
python ACE/preprocess/preprocess_data.py
```

This generates cache files containing both the preprocessed audio features and chord labels. Parameters for the preprocessing script can be adjusted in `ACE/preprocess/dataset.gin`.

## 🚀 Training
To train a model, run:

```bash
python -m ACE.trainer --model model_name --name run_name
```
* `--model`: choose between
    * `conformer`: baseline classification model 
    * `conformer_decomposed`: decomposition-based proposed in the paper. 

* `--name`: optional, used for logging the run on [Weights & Biases](https://wandb.ai/).

Parameters for the training script can be adjusted in `ACE/trainer.gin`.

## 🎯 Models
Two models are implemented in this repository:

* `conformer`: baseline architecture for chord classification (170 classes).
* `conformer_decomposed`: our proposed model, predicting root, bass, and pitch activations separately.

The `conformer_decomposed` model introduces several key innovations:

1. **Decomposed Output Heads**: Instead of a single output layer for all chord classes, we use separate heads for root, bass, and chord tones, allowing for more specialized learning.
2. **Consonance Label Smoothing**: adds music-theory informed smoothing, improving robustness and harmonic plausibility.

Models are stored in the `ACE/models` directory, which also contains `.gin` configuration files for each model.

## 🔮 Inference

Run chord inference on an audio file using the trained **conformer_decomposed** model:

```bash
python -m ACE.inference --audio path/to/audio.wav --out path/to/output.lab --chord-min-duration 0.5
```

The `--chord-min-duration` parameter sets the minimum duration (in seconds) for chord segments; shorter segments will be merged with adjacent ones.

This processes the entire track in 20 s segments, decodes predictions, merges identical consecutive chords, and outputs a .lab file:

```
start_time    end_time    chord_label
0.000000      2.581995    E:maj
2.581995      5.163990    B:maj
5.163990      7.745985    A:maj
```

The resulting .lab file contains the full chord sequence for the input track and can be visualised in Sonic Visualiser or any standard chord annotation tool.

By default, inference uses the pretrained checkpoint located at
`ACE/checkpoints/conformer_decomposed_smooth.ckpt`, provided in this repository.

**Optional parameters:**

- `--model-name`: Model to use for inference. Choices: `conformer` or `conformer_decomposed` (default: `conformer_decomposed`).

- `--threshold`: Threshold for chord component activation (only used with `conformer_decomposed`). Type: `float` (default: `0.5`). 

- `--chunk-dur`: Duration of audio chunks to process (in seconds). Type: `float` (default: `20.0`). Note: This cannot be changed when `conformer_decomposed` model is used with `consonance_decomposed` loss.

- `--beat-sync`: If set, predicts one chord per beat instead of removing short chord segments. Beats are estimated with [Beat This!](https://github.com/CPJKU/beat_this), and each beat is assigned the chord that covers the largest portion of that beat (i.e. the most present chord), reducing fragmentation without arbitrary duration thresholds. When enabled, `--chord-min-duration` is ignored.

- `--beat-checkpoint`: Beat This! checkpoint to use for beat tracking (only relevant with `--beat-sync`). Type: `str` (default: `final0`).


## 📑 Citation

If you use this code, please cite:

```bibtex 
@inproceedings{poltronieri2025discord,
  title     = {From Discord to Harmony: Decomposed Consonance-Based Training for Improved Audio Chord Estimation},
  author    = {Poltronieri, Andrea and Serra, Xavier and Rocamora, Martín},
  booktitle = {Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  publisher = {International Society for Music Information Retrieval, ISMIR 2025},
  year      = {2025},
  location  = {Daejeon, South Korea},
  url={https://arxiv.org/abs/2509.01588}, 
}
```

## 📜 License
MIT License

Copyright (c) 2025 Andrea Poltronieri, Xavier Serra, Martín Rocamora

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
