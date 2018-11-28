# AlexNet in Tensorflow

## Run

- Train on CIFAR-10

```shell
$ python -m alexnet.main
```

## Benchmark

|                  |CIFAR10  |CIFAR10-augment  |
|------------------|:-------:|:---------------:|
|AlexNet-3         |0.7489   |0.7952           |
|AlexNet-3+LRN     |0.7486   |0.7925           |
|AlexNet-4         |0.7558   |0.8178           |
|AlexNet-4+LRN     |0.7452   |0.8150           |
|AlexNet-3+OP      |0.7664   |0.7848           |