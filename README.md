## HiHPQ

This repo includes the official implementation of  **HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval**.



#### Reproduction

Codes are tested on the following python environment.

```python
python  3.9.13
torch  1.12.1
scikit-learn  1.0.2
numpy 1.26.4
```

Please refer to the `./scripts/` directory to reproduce the main results.  An example run is:

```python
sh ./scripts/cifar10_ii/16bits.sh
```

Note that for experiments on `NUS-WIDE` and `Flickr`, you should first download the raw datasets.  You can find the download information in [this page](https://github.com/gimpong/AAAI22-MeCoQ).



#### Acknowledgements

The code implementation is based on the [MeCoQ](https://github.com/gimpong/AAAI22-MeCoQ), [HyboNet](https://github.com/chenweize1998/fully-hyperbolic-nn) and [HCNN](https://github.com/kschwethelm/HyperbolicCV).



#### Reference

If you find this code useful in your research, please cite the following paper:

```
@inproceedings{qiu2024hihpq,
  title={HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval},
  author={Qiu, Zexuan and Liu, Jiahong and Chen, Yankai and King, Irwin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}

```
