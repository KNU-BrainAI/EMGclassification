# EMGhandnet

Before starting our project, we tested previous study's model.   
This is just for testing previous study's model and dataset, this code may not perfectly work.   
And it can be different from original paper because we couldn't find original source code and that made with tensorflow, so we just made model's architecture with pytorch and limitted information of model.   
But without some information and preprocessing, it performed almost same as original one's performance at NinaPro db1 dataset.   


## Environment
Python 3.8
Pytorch 1.12.1
Pandas 1.4.3

```python
example
python train.py --batch-size 16 --epochs 200 --learning-rate 1e-4
```

## Reference
[EMGHandNet: A hybrid CNN and Bi-LSTM architecture for hand activity classification using surface EMG signals]
https://www.sciencedirect.com/science/article/pii/S0208521622000080
[NinaPro datasets]
http://ninaweb.hevs.ch/
