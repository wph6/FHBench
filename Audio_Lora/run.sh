python wav2vec_lora_fed.py --dataset SPRS --datapercent 0.3 --fed_alg fedavg   --lr 3e-2
python wav2vec_lora_fed.py --dataset SPRS --datapercent 0.3 --fed_alg fedprox  --lr 3e-2
python wav2vec_lora_fed.py --dataset SPRS --datapercent 0.3 --fed_alg SCAFFOLD --lr 3e-2
python wav2vec_lora_fed.py --dataset SPRS --datapercent 0.3 --fed_alg APFL     --lr 3e-2
python wav2vec_lora_fed.py --dataset SPRS --datapercent 0.3 --fed_alg APPLE    --lr 3e-2
python wav2vec_lora_fed.py --dataset SPRS --datapercent 0.3 --fed_alg fedALA   --lr 3e-2

python wav2vec_lora_fed.py --dataset ICBHI --datapercent 0.3 --fed_alg fedavg   --lr 1e-1
python wav2vec_lora_fed.py --dataset ICBHI --datapercent 0.3 --fed_alg fedprox  --lr 1e-1
python wav2vec_lora_fed.py --dataset ICBHI --datapercent 0.3 --fed_alg SCAFFOLD --lr 1e-1
python wav2vec_lora_fed.py --dataset ICBHI --datapercent 0.3 --fed_alg APFL     --lr 1e-1
python wav2vec_lora_fed.py --dataset ICBHI --datapercent 0.3 --fed_alg APPLE    --lr 1e-1
python wav2vec_lora_fed.py --dataset ICBHI --datapercent 0.3 --fed_alg fedALA   --lr 1e-1
