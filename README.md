# ERDNS
 

## Dependencies
* Python 3.9
* PyTorch 1.8
* Numpy 1.20
* Scipy 1.6



## Usage

To run conventional models and ConvE model.

bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -s 0.5
bash run.sh train TransE wn18rr 0 0 512 512 500 6.0 0.5 0.00005 80000 8 -s 1
bash run.sh train TransE YAGO3-10 0 0 1024 1024 500 6.0 0.5 0.0002 200000 8 -s 1

bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de -s 0.5
bash run.sh train RotatE wn18rr 0 0 512 128 500 6.0 0.5 0.00005 80000 8 -de -s 1
bash run.sh train RotatE YAGO3-10 0 0 1024 1024 500 6.0 0.5 0.0002 200000 8 -de -s 1

bash run.sh train DistMult FB15k-237 0 0 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001 -s 1 --ignore_scoring_margin
bash run.sh train DistMult wn18rr 0 0 512 50 1000 200.0 1.0 0.002 80000 8 -r 0.000005 -s 1 --ignore_scoring_margin
bash run.sh train DistMult YAGO3-10 0 0 1024 512 1000 200.0 1.0 0.001 200000 8 -r 0.000002 -s 1 --ignore_scoring_margin

bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001 -s 1 --ignore_scoring_margin
bash run.sh train ComplEx wn18rr 0 0 512 126 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005 -s 1 --ignore_scoring_margin
bash run.sh train ComplEx YAGO3-10 0 0 1024 512 500 200.0 1.0 0.001 200000 8 -de -dr -r 0.000002 -s 1 --ignore_scoring_margin

bash run.sh train ConvE FB15k-237 0 0 1024 4096 200 200.0 1.0 0.0001 100000 16 -r 0.005 -s 1  --ignore_scoring_margin --conve_drop0=0.1 --conve_drop1=0.3 --conve_drop2=0.3
bash run.sh train ConvE wn18rr 0 0 512 4096 200 200.0 1.0 0.0005 80000 16 -r 0.001 -s 1 --ignore_scoring_margin --conve_drop0=0.2 --conve_drop1=0.2 --conve_drop2=0.2


To run CompGCN model.

Go into OurCompGCN

python run.py -score_func transe -opn mult -gamma 9 -hid_drop 0.2 -init_dim 200 -data WN18RR -square 1 -N_sample 1024 -batch 512
python run.py -score_func transe -opn mult -gamma 9 -hid_drop 0.2 -init_dim 200 -data FB15k-237 -square 1 -N_sample 4096 -batch 1024

python run.py -score_func distmult -opn mult -gcn_dim 150 -gcn_layer 2 -data WN18RR -square 1 -N_sample 8192 -batch 512
python run.py -score_func distmult -opn mult -gcn_dim 150 -gcn_layer 2 -data FB15k-237 -square 1 -N_sample 4096 -batch 1024

python run.py -score_func conve -opn mult -square 1 -data WN18RR -square 1 -N_sample 4096 -batch 512
python run.py -score_func conve -opn mult -square 1 -data WN18RR -square 1 -N_sample 4096 -batch 1024

## Data
We provide three datasets: FB15K237, WN18RR and YAGO3-10.

