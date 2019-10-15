# CARAFE_CUDA

This repo is a PyTorch cuda extension of the paper ["CARAFE: Content-Aware ReAssembly of FEatures"](https://arxiv.org/abs/1905.02188).
It can replace the upsamling layer in the networks and is easy to be integrated into [mmdetection](https://github.com/open-mmlab/mmdetection).

## Comparison with plain PyTorch

CARAFE can be implemented in plain PyTorch code by `torch.nn.functional.unfold`. However, the `unfold` function reassembles the features within a local region to
generate a large feature map, which causes memory problem and is slow when the input is large.

CARAFE cuda extension can solve the memory problem and is **~10x** faster than the code in plain PyTorch. More details can be referred to [test.py](test.py).

## How To Use

* Keep the environment to be consistent with mmdetection.
* Build

```shell
cd carafe_layer/
python setup.py build_ext --inplace
```
* Test
```shell
python test.py
```

## Results

We replace the upsampling layers with [carafe_module](carafe_module.py) in Mask R-CNN. The results with ResNet-50 are list as follows:

| Method | Orig-paper(box AP / mask AP) | This-repo(box AP / mask AP) |
| ------ |------ | ------ |
| Baseline | 37.4/34.2 | 37.3/34.2 |
| FPN w/ CARAFE | 38.6/35.2 | 37.9/34.6 |
| FPN + M.H w/ CARAFE | 38.6/35.7 | -/- |

NOTE: `M.H` represents `Mask Head` in Mask R-CNN.

More results will be added.
