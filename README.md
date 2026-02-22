### Setup
Use conda

Create environment
```
conda create -n vibrio_lstm python=3.10
conda activate vibrio_lstm
```
Install PyTorch (GPU)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install PyTorch (CPU)
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Requirements
```
conda install pandas numpy
```

Run the scripts
```
python train.py
```
