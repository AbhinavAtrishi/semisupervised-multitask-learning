# Semisupervised Multitask Learning

This repository is an unofficial and slightly modified implementation of [UM-Adapt](https://arxiv.org/abs/1908.03884 "UM-Adapt: Unsupervised Multi-Task Adaptation Using Adversarial Cross-Task Distillation")[1] using PyTorch.

This code primarily deals with the tasks of sematic segmentation, instance segmentation, depth prediction learned in a multi-task setting (with a shared encoder) on a synthetic dataset and then adapted to another dataset with a domain shift. Specifically for this implementation the aim is to learn the three tasks on the [Cityscapes Dataset](https://www.cityscapes-dataset.com), then adapt and evaluate performance in a fully unsupervised or a semi-supervised setting on the [IDD Dataset](https://idd.insaan.iiit.ac.in/ "India Driving Dataset").

The architecture used for the semantic and instance segmentation model is taken from [Panoptic Deeplab](https://arxiv.org/abs/1911.10194 "Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation")[2]. While a choice for the depth decoder is offered between [BTS](https://arxiv.org/abs/1907.10326 "From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation")[3] and [FCRN-Depth](https://arxiv.org/abs/1606.00373 "Deeper Depth Prediction with Fully Convolutional Residual Networks")[4].

## Usage
The following commands can be used to run the codebase, please make sure to see the respective papers for more details.

1. To train the base encoder on the Cityscapes (or any other dataset with appropriate modifications) use the following command. Additional flags can also be set as required:

    `python base_trainer.py --name BaseRun --cityscapes_dir /path/to/cityscapes`

2. Then train the CCR Regularizer as proposed in UM-Adapt with the following command:

    `python ccr_trainer.py --base_name BaseRun --cityscapes_dir /path/to/cityscapes --hed_path /path/to/pretrained/HED-Network`

3. Unsupervised adaptation to IDD can now be performed using:

    `python idd_adapter.py --name AdaptIDD --base_name BaseRun --cityscapes_dir /path/to/cityscapes --idd_dir /path/to/idd --hed_path /path/to/pretrained/HED-Network`

4. Further optional semi-supervised fine-tuning can be done using:

    `python idd_supervised.py --name SupervisedIDD --base_name BaseRun --idd_name AdaptIDD --idd_epoch 10 --idd_dir /path/to/idd --hed_path /path/to/pretrained/HED-Network --supervised_pct 0.5`

The code can generally be modified to suit any dataset as required, the base architectures of different decoders as well as the shared encoders can also be altered as needed. 

## References

If you find this code helpful in your research, please consider citing the following papers.

```
[1]  @inproceedings{Kundu_2019_ICCV,
        author = {Kundu, Jogendra Nath and Lakkakula, Nishank and Babu, R. Venkatesh},
        title = {UM-Adapt: Unsupervised Multi-Task Adaptation Using Adversarial Cross-Task Distillation},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month = {October},
        year = {2019}
    }
```

```
[2]  @inproceedings{cheng2020panoptic,
        author={Cheng, Bowen and Collins, Maxwell D and Zhu, Yukun and Liu, Ting and Huang, Thomas S and Adam, Hartwig and Chen, Liang-Chieh},
        title={Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2020}
    }
```

```
[3]  @article{lee2019big,
        title={From big to small: Multi-scale local planar guidance for monocular depth estimation},
        author={Lee, Jin Han and Han, Myung-Kyu and Ko, Dong Wook and Suh, Il Hong},
        journal={arXiv preprint arXiv:1907.10326},
        year={2019}
}
```

```
[4]  @inproceedings{Xie_ICCV_2015,
         author = {Saining Xie and Zhuowen Tu},
         title = {Holistically-Nested Edge Detection},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2015}
     }
```

```
[5]  @misc{pytorch-hed,
         author = {Simon Niklaus},
         title = {A Reimplementation of {HED} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-hed}}
    }
```

If you use either of Cityscapes or IDD datasets, consider citing them

```
@inproceedings{Cordts2016Cityscapes,
    title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
    author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
    booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016}
}
```

```
@article{DBLP:journals/corr/abs-1811-10200,,
    title={IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments},
    author = {Varma, Girish and Subramanian, Anbumani and Namboodiri, Anoop and Chandraker, Manmohan and Jawahar, C.V.}
    journal={arXiv preprint arXiv:1811.10200},
    year={2018}
```

Finally, if you use the Xception backbone, please consider citing

```
@inproceedings{deeplabv3plus2018,
    title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
    author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
    booktitle={ECCV},
    year={2018}
}
```


## Acknowledgements

Utility functions from many wonderful open-source projects were used, I would like to especially thank the authors of:

* [Panoptic-Deeplab](https://github.com/bowenc0221/panoptic-deeplab)
* [Pytorch-HED](https://github.com/sniklaus/pytorch-hed)
* [FCRN_Pytorch](https://github.com/dontLoveBugs/FCRN_pytorch)
* [BTS](https://github.com/cogaplex-bts/bts/tree/master/pytorch)
* [Detectron2](https://github.com/facebookresearch/detectron2)
* [TorchVision](https://github.com/pytorch/vision)