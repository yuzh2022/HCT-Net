# HCT-Net

**This repository is an official implementation of the paper  HCT-net: hybrid CNN-transformer model based on a neural architecture search network for medical image segmentation.**

`Yu Z, Lee F, Chen Q. HCT-net: hybrid CNN-transformer model based on a neural architecture search network for medical image segmentation[J]. Applied Intelligence, 2023: 1-17.`

Considering that many manually designed convolutional neural networks (CNNs) for different tasks that require considerable time, labor, and domain knowledge have been designed in the medical image segmentation domain and that most CNN networks only consider local feature information while ignoring the global receptive field due to the convolution limitation, there is still much room for performance improvement. Therefore, designing a new method that can fully capture feature information and save considerable time and human energy with less GPU memory consumption and complexity is necessary. In this paper, we propose a novel hybrid CNN-transformer model based on a neural architecture search network (HCT-Net), which designs a hybrid U-shaped CNN with a key-sampling Transformer backbone that considers contextual and long-range pixel information in the search space and uses a single-path neural architecture search that contains a flexible search space and an efficient search strategy to simultaneously find the optimal subnetwork including three types of cells during SuperNet. Compared with various types of medical image segmentation methods, our framework can achieve competitive precision and efficiency on various datasets, and we also validate the generalization on unseen datasets in extended experiments. In this way, we can verify that our method is competitive and robust.



## Preprocess

### Dataset

1. CVC-ClinicDB 
2. CHAOS-CT
3. ISIC-2018

## Environment

- Python = 3.9

- Pytorch = 1.13.1

- TensorboardX

- pydicom

- At least 16GB-Tesla T4 GPU (batch size = 4)

  ....

## Training

**Start Training**

- for example

```python
python hct_net/train_CVCDataset.py
```

- Then, find the `model_best.pth.rar` in correspond `search_exp` folder and write `the final genotypes` of log into `genotypes.py` with your custom name.

## Retrain

- Put the path of `model_best.pth.rar` and the final genotypes name into `retrain_cvcDataset`

  ```python
  python hct_net/retrain_cvcDataset.py
  ```

- Then, find the `model_best.pth.rar` in correspond `logs` folder

## Test

Put the path of `model_best.pth.rar` and the final genotypes name into `test_cvc.py`

```python
python hct_net/test_cvc.py
```

## References

[1] :  [Thanks for [MixSearch: Searching for Domain Generalized Medical Image Segmentation Architectures](https://github.com/lswzjuer/NAS-WDAN.git)

[2] :  [Thanks for  [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://github.com/fundamentalvision/Deformable-DETR.git)


