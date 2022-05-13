```
python convert_dataset_tocsv.py
```

```
brainome -headerless -nosplit -vv cifar_train.csv -o cifar_out_rf_vv.py
```

## How to run our ResNet18 model on CIFAR-10
```
python train.py --final_dim 100
```

Ref: https://github.com/kuangliu/pytorch-cifar
