## ATFM

[AAAI-26 Oral] Official pytorch implementation of 'Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation'


## Introduction
A simultaneous enhancement of accuracy and diversity of predictions remains a challenge in ambiguous medical image segmentation (AMIS) due to the inherent trade-offs. While truncated diffusion probabilistic models (TDPMs) hold strong potential with a paradigm optimization, existing TDPMs suffer from entangled accuracy and diversity of predictions with insufficient fidelity and plausibility. To address the aforementioned challenges, we propose Ambiguity-aware Truncated Flow Matching (ATFM), which introduces a novel inference paradigm and dedicated model components. Firstly, we propose Data-Hierarchical Inference, a redefinition of AMIS-specific inference paradigm, which enhances accuracy and diversity at data-distribution and data-sample level, respectively, for an effective disentanglement. Secondly, Gaussian Truncation Representation (GTR) is introduced to enhance both fidelity of predictions and reliability of truncation distribution, by explicitly modeling it as a Gaussian distribution at $T_{\text{trunc}}$ instead of using sampling-based approximations. Thirdly, Segmentation Flow Matching (SFM) is proposed to enhance the plausibility of diverse predictions by extending semantic-aware flow transformation in Flow Matching (FM). Comprehensive evaluations on LIDC and ISIC3 datasets demonstrate that ATFM outperforms SOTA methods and simultaneously achieves a more efficient inference. ATFM improves GED and HM-IoU by up to $12\%$ and $7.3\%$ compared to advanced methods.

![Overview of proposed ATFM](/fig/main.png)

## Requirements
You can build the dependencies by executing the following command
'''
conda create -n FM python=3.9
source activate FM
pip install -r requirement.txt
'''

## Dataset
Two public datasets: LIDC and ISIC Subset are implemented in this work. You can download the datasets in the following links:
- [LIDC Dataset](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5) as preprocessed by @Stefan Knegt
- [ISIC Subset](https://drive.google.com/file/d/1m7FdNldGqGyqw2L9GX8HDrHDId3kExtH/view?usp=sharing) as preprocessed by @killanzepf

The corresponding paths of the datasets should be modified in `metadata_managr.py`

## Training Procudure
- Step 1: Train Gaussian Truncation Representation for both datasets
  '''
  python train_GTR.py --what LIDC --epochs 1000 --betchsize 256 --save_model True --save_model_step 50
  python train_GTR.py --what isic3_style_concat --epochs 400 --batchsize 8 --save_model True --save_model_step 50
  '''
- Step 2: Train Segmentation Flow Matching based on the frozen GTR
  '''
  python train_prior_LIDC.py
  python train_prior_ISIC.py
  '''

Note: The GPU memory consumption of ISIC Subset is 24198MiB (23.63GiB) even with a batchsize = 1. Therefore, CUDA may appear to be out-of-memory when too much memory is reserved. Feel free to use `torch.utils.checkpoint` to reduce the GPU memory comsumption.
'''
pred = model(image) # Original forward
pred = torch.utils.checkpoint(model, image) # Forward with lower memory comsumption
'''

## Testing Procedure
'''
python test_prior_LIDC.py
python test_prior_ISIC.py
'''

By visualizing the predictions, you can obtain a series of predictions with both high accuracy and diversity, while the fidelity and plausibility is enhanced simultaneously.
![LIDC Dataset](/fig/pred.png)
![ISIC Subset](/fig/pred1.png)

## Acknowledgements
- We thank [@killanzepf](https://github.com/kilianzepf/conditioned_uncertain_segmentation) for the preprocessed dataset and GTR baseline.
- We thank [@aleksandrinvictor](https://github.com/aleksandrinvictor/flow-matching) for the Flow Matching baseline.
- We thank [@Stefan Knegt](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) for the preprocessed dataset.

## Citations
You can cite this paper with the following bibtex code if you find this work helpful:
'''
@article{li2025ambiguity,
  title={Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation},
  author={Li, Fanding and Li, Xiangyu and Su, Xianghe and Qiu, Xingyu and Dong, Suyu and Wang, Wei and Wang, Kuanquan and Luo, Gongning and Li, Shuo},
  journal={arXiv preprint arXiv:2511.06857},
  year={2025}
}
'''
