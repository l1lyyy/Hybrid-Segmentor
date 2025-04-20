
# Hybrid-Segmentor – Crack Segmentation via CNN/Transformer Paths

This project implements a hybrid crack segmentation architecture using both CNN and Transformer branches. Additionally, it supports independent training of each path to investigate their individual contributions.

---

##  Project Structure

```
Hybrid_Segmentor/
├── trainer.py                # Train Hybrid (CNN + Transformer)
├── trainer_cnn.py           # Train CNN-only model
├── trainer_transformer.py   # Train Transformer-only model
├── model.py                 # Model definition with hybrid and ablation options
├── config.py                # Path & hyperparameter configuration
├── dataloader.py            # Augmentation and loading
├── test.py                  # Inference script for test set
├── metric.py                # Evaluation metrics
├── callback.py              # Callbacks for logging and saving
├── utils.py                 # Helper functions
```

---

## Dataset: CrackVision12K

Ensure your dataset is structured as follows:

```
dataset/
├── train/
│   ├── IMG/
│   └── GT/
├── val/
│   ├── IMG/
│   └── GT/
└── test/
    ├── IMG/
    └── GT/
```

Update `config.py` with appropriate paths.

---

## Training

Train the full Hybrid model:

```bash
python Hybrid_Segmentor/trainer.py
```

Train CNN-only:

```bash
python Hybrid_Segmentor/trainer_cnn.py
```

Train Transformer-only:

```bash
python Hybrid_Segmentor/trainer_transformer.py
```

---

## Evaluation

Evaluate any trained model on the test set:

```bash
python Hybrid_Segmentor/test.py
```

Modify the following variables inside `test.py`:
```python
CHECKPOINT_PATH = "path/to/model.ckpt"
OUTPUT_PATH = "path/to/save/outputs"
```

---

## Visualization

Sample predictions are stored in `/outputs/` after running `test.py`. They can be visualized using standard image viewers.

---

## Citation

If you use this work, please cite:

```
@article{goo2024hybridsegmentor,
  title={Hybrid-Segmentor: A Hybrid Approach to Automated Fine-Grained Crack Segmentation in Civil Infrastructure},
  author={Goo, June Moh and Milidonis, Xenios and Artusi, Alessandro and Boehm, Jan and Ciliberto, Carlo},
  journal={arXiv preprint arXiv:2409.02866},
  year={2024}
}
```

---

## Maintainer

- Project by: Trang Nhựt – University of Science (HCMUS)
- Notebooks:
  - [CNN Branch](https://www.kaggle.com/code/nhuttrang/cnn-branch)
  - [Transformer Branch](https://www.kaggle.com/code/nhuttrang/transformer-branch)
