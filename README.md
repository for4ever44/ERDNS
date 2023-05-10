# ERDNS
 

## Dependencies
* Python 3.9
* PyTorch 1.8
* Numpy 1.20
* Scipy 1.6



## Usage
python train.py --task_dir=./KG_Data/WN18RR --model=TransE  --margin=8 --bern=1 --out_file=_test --topk=1000 --gpu=1 --N_1=50 --N=100 --hidden_dim=200 --lr=0.0001 --n_epoch=1000 --n_batch=1024 --filter=True --epoch_per_test=100 --test_batch_size=20 --optim=adam;
python train.py --task_dir=./KG_Data/FB15K237 --model=DistMult  --margin=0 --bern=1 --out_file=_test --topk=1000 --gpu=1 --N_1=50 --N=100 --hidden_dim=200 --lr=0.0001 --n_epoch=1000 --n_batch=1024 --filter=True --epoch_per_test=100 --test_batch_size=20 --optim=adam;


## Data
We provide three datasets: FB15K237, WN18RR and YAGO3-10.

