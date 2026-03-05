# YOWOv3-Improved: An Improved and Extended codebase of original [YOWOv3](https://github.com/hope1337/YOWOv3) for Human Action Detection and Recognition

This repository is an improved and extended version of [YOWOv3](https://github.com/AakiraOtok/YOWOv3), originally proposed in the paper. The main aim is to make cleaner and more extensible codebase:

> **YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition**
> Duc Manh Nguyen Dang, Viet Hang Duong, Jia Ching Wang, Nhan Bui Duc
> [arXiv:2408.02623](https://arxiv.org/abs/2408.02623)

We were asked to do a pull request to the orignal repo in the one of the official repo's [issue](https://github.com/hope1337/YOWOv3/issues/61#issuecomment-3314445030) but due to the extent of structural changes made to the codebase, a direct pull request was not practical — however, the offer remains open if the original author wishes to incorporate any improvements.

> **Note:** This code has been used in real-world projects and is working well. However, it may still contain limitations or mistakes in the modified version of the code. If you find any issues or have suggestions, feel free to open a discussion or issue — I'm happy to talk about it.

---

## What's new in YOWOv3-Improved and planned to add in the future

Compared to the original [YOWOv3](https://github.com/hope1337/YOWOv3) codebase, this repository introduces the following improvements:

- [x]  **DDP (Distributed Data Parallel)** — Multi-GPU training support for faster and more scalable training.
- [x]  **Custom Augmentation Pipeline** — Flexible and extensible data augmentation using pure openCV to improve model robustness and generalization.
- [x]  **Validation Logic** — Proper validation loop integrated into training, allowing you to monitor performance on a held-out set during training.
- [x]  **Updated TAL Loss** — Modified TAL (Task-Aligned Learning) loss to help the model learn from background images and significantly reduce false positives in predictions. However, I could not touch SimOTA loss because the updating TAL was just enogh for me. 
- [x]  **Cleaner and More Extensible Codebase** — The overall structure has been reorganized to make it easier to plug in custom datasets, backbones, and other components.
- [ ] **Rectangle input size** (working on) — Extending model input size to different input resolutions (e.g., 384x640) which is very beneficial in Surveiallnce systems (right now, model supports square input size e.g., 640x640)
- [ ] **Reproducing 3DCNNs results** (working on) — Working on reproducing [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs) to make codebase reusable and improve accuracy (hopefully...). This will be posted in a seperate repo and will be announced here.
---

## Dataset Support

This repository primarily supports the **UCF101-24** dataset structure. However, you can easily extend it to use other datasets with minimal effort.

### UCF101-24
Download from (as in YOWOv2):
```
https://drive.google.com/file/d/1Dwh90pRi7uGkH5qLRjQIFiEmMJrAog5J/view

```

---

## Preparation

### Environment Setup

Clone this repository:
```bash
git clone https://github.com/dilwolf/YOWOv3-Improved.git
cd YOWOv3-Improved
```

Install the dependencies (tested Python 3.10+):
```bash
pip install -r requirements.txt
```
---
## Basic Usage

### Config File

Almost all configurations are controlled through a config file as in the original [YOWOv3](https://github.com/hope1337/YOWOv3). Sample config files are provided in the `util/config.md` file.

> **Warning:** All options must remain consistent between training and inference. Changing settings like `image_size` or `backbone3D` between training and testing will severely degrade performance.

### Training/Multi-GPU Training with DDP
```bash
torchrun --nproc_per_node=$ main.py --train
```
Replace `$` with the number of GPUs available on your machine.

### Simply run for a single GPU:
```bash
python main.py --train
```

### Evaluation

```bash
torchrun --nproc_per_node=2 main.py --valid
```

### Detection (result saved as a video file)

```bash
python detect.py --weight best.pth --source path/to/source
```

### Export to ONNX

```bash
python export_onnx.py
```
---

## Pretrained Resources

Pretrained backbone and model checkpoints from the original YOWOv3 are available on the author's Hugging Face repository:

```
https://huggingface.co/manh6054/YOWOv3/tree/main
```

These checkpoints are compatible with this repository, provided that the config options are aligned correctly.

---

## Limitations and Known Issues

- Some edge cases in augmentation may not be fully handled — contributions and feedback are welcome.
- AVA v2.2 evaluation is not the primary focus of this repo but may still work with minor adjustments.
- The codebase has been validated on real-world projects but may contain bugs in less common configurations.

If you encounter any issues or have insights to share, please open an [issue](../../issues) or start a [discussion](../../discussions).

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, consistent with the upstream dependencies this project relies on. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this repository in your research, please consider citing the original YOWOv3 paper:

```bibtex
@misc{dang2024yowov3efficientgeneralizedframework,
      title={YOWOv3: An Efficient and Generalized Framework for Human Action Detection and Recognition}, 
      author={Duc Manh Nguyen Dang and Viet Hang Duong and Jia Ching Wang and Nhan Bui Duc},
      year={2024},
      eprint={2408.02623},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02623}, 
}
```

---

## References and Acknowledgements

This project builds upon the following repositories, and I am deeply grateful to their authors:

- [YOWOv3](https://github.com/hope1337/YOWOv3) — the original codebase this repo extends
- [YOWOv2](https://github.com/yjh0410/YOWOv2) — the this repo was used in the original codebase
- [YOWO](https://github.com/wei-tim/YOWO)  — the first YOWO series
- [YOLOv8-pt](https://github.com/jahongir7174/YOLOv8-pt) — neat YOLOv8 PyTorch implementation
- [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs) — 3D CNN backbones
- [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) — I3D model implementation
- [AVAv2.2 Evaluation](https://github.com/activitynet/ActivityNet/tree/master/Evaluation) — official evaluation code