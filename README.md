# Advanced Topics in Audio Processing using Deep Learning - Final Project

This repository contains our implementation for the final project in **Advanced Topics in Audio Processing using Deep Learning**.  
Our work focuses on training and evaluating multiple **TinySpeech** model architectures for audio classification using the **SpeechCommands** dataset.   

---

## Project Overview

- **Models:**  
  We implement four model architectures: 
  - TinySpeechX
  - TinySpeechY
  - TinySpeechZ
  - TinySpeechM  

- **Evaluation:**  
  An evaluation script allows you to either test a specified audio file (with an optional true label) or randomly sample 10 recordings from the dataset for inference.

--- 

## 🛠️ How to Setup the Environment
We worked with Torch version 2.6.0 with CUDA version 12.6 and trained the models on local NVIDIA RTX-3060 (12GB).
Thus, Torch (with cuda support) is needed to be manually install into the project interpreter by running the following:
```bash
pip install torch==2.6.0+cu126 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
```
Next, Please run:
```bash
pip install -r requirements.txt
```
---
## 🔍 Run All Evaluations at Once *(recommended)*
To evaluate all trained models in sequence on 10 random samples (Powershell or Pycharm), run the following command:
```bash
python main.py eval --weight models/weights/TinySpeechX_85.pth --model TinySpeechX && \
python main.py eval --weight models/weights/TinySpeechY_84.pth --model TinySpeechY && \
python main.py eval --weight models/weights/TinySpeechZ_82.pth --model TinySpeechZ && \
python main.py eval --weight models/weights/TinySpeechM_81.pth --model TinySpeechM
```
Otherwise, run them one by one.

---

## Train the Models:
  ### With Deterministic Seed (default):
  ```bash
  python main.py train --name <run_name>
  ```
  ###  Without Seed (Nondeterministic Mode):
  ```bash
  python main.py train --no-seed --name <run_name>
  ```
---

## Evaluate a Model:
  Required Arguments:
- --weight: Path to a model weights (.pth file, locate in this repo at /models/weights/<model_name>)
- --model: Model architecture to use (TinySpeechX, TinySpeechY, TinySpeechZ, TinySpeechM).

### Evaluate on Random Dataset Samples:

```bash
python main.py eval --weights /models/weights/<model_name> --model <model_name>
```

### Evaluate a Specific Audio File:
```bash
python main.py eval --weights /models/weights/<model_name> --model <model_name> --file path/to/audio.wav
```
