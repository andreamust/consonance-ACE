# 🎶 **From Discord to Harmony**: Decomposed Consonance-Based Training for Improved Audio Chord Estimation

This repository contains the official implementation of the paper:  

> **From Discord to Harmony: Decomposed Consonance-Based Training for Improved Audio Chord Estimation**  
> Andrea Poltronieri, Xavier Serra, Martín Rocamora  
> *Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR), Daejeon, South Korea, 2025.*  


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
python ACE/preprocess_data.py
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

## 🎯 Models
Two models are implemented in this repository:

* `conformer`: baseline architecture for chord classification (170 classes).
* `conformer_decomposed`: our proposed model, predicting root, bass, and pitch activations separately.

The `conformer_decomposed` model introduces several key innovations:

1. **Decomposed Output Heads**: Instead of a single output layer for all chord classes, we use separate heads for root, bass, and chord tones, allowing for more specialized learning.
2. **Consonance Label Smoothing**: adds music-theory informed smoothing, improving robustness and harmonic plausibility.

Models are stored in the `ACE/models` directory, which also contains `.gin` configuration files for each model.

## 🔮 Inference

TBD

<!-- To run inference with a trained model, use the following command:

```bash
python -m ACE.inference --model model_name --input audio_file
```

* `model_name`: specify the model to use (either `conformer` or `conformer_decomposed`).
* `audio_file`: path to the audio file for which to predict chord labels. -->

## 📑 Citation

If you use this code, please cite:

```bibtex 
@inproceedings{poltronieri2025discord,
  title     = {From Discord to Harmony: Decomposed Consonance-Based Training for Improved Audio Chord Estimation},
  author    = {Poltronieri, Andrea and Serra, Xavier and Rocamora, Martín},
  booktitle = {Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR)},
  year      = {2025},
  location  = {Daejeon, South Korea},
  publisher = {International Society for Music Information Retrieval},

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
