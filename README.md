# FedNN
Pattern Recognition. "FedNN: Federated learning on concept drift data using weight and adaptive group normalizations"

# Train

    # FedNN
    python main.py \
        --method fedavg \
        --data_dir ./data \
        --dataset_name DIGIT \
        --model_name LeNet_fednn \
        --result_path results

    # naive FedAvg
    python main.py \
        --method fedavg \
        --data_dir ./data \
        --dataset_name DIGIT \
        --model_name LeNet \
        --result_path results

# Data
Please download the pre-processed digit datasets here [FedBN](https://github.com/med-air/FedBN) and put it in the ./data directory.

# Requirements

    pip install torch torchvision

# Citation
If you find this repository useful in your research, please cite:
```
@article{kang2023fednn,
  title={FedNN: Federated learning on concept drift data using weight and adaptive group normalizations},
  author={Kang, Myeongkyun and Kim, Soopil and Jin, Kyong Hwan and Adeli, Ehsan and Pohl, Kilian M and Park, Sang Hyun},
  journal={Pattern Recognition},
  volume={149},
  pages={110230},
  year={2023},
  publisher={Elsevier}
}
```

Thanks to works below for their implementations which were useful for this work.
[FedDC](https://github.com/gaoliang13/FedDC)
