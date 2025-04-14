python llm_lora_fed.py --dataset med_abs --datapercent 0.5 --fed_alg fedavg   --lr 8e-3
python llm_lora_fed.py --dataset med_abs --datapercent 0.5 --fed_alg fedprox  --lr 8e-3
python llm_lora_fed.py --dataset med_abs --datapercent 0.5 --fed_alg SCAFFOLD --lr 8e-3
python llm_lora_fed.py --dataset med_abs --datapercent 0.5 --fed_alg APFL     --lr 8e-3
python llm_lora_fed.py --dataset med_abs --datapercent 0.5 --fed_alg APPLE    --lr 8e-3
python llm_lora_fed.py --dataset med_abs --datapercent 0.5 --fed_alg fedALA   --lr 8e-3

python llm_lora_fed.py --dataset pubmed --datapercent 0.005 --fed_alg fedavg   --lr 5e-3
python llm_lora_fed.py --dataset pubmed --datapercent 0.005 --fed_alg fedprox  --lr 5e-3
python llm_lora_fed.py --dataset pubmed --datapercent 0.005 --fed_alg SCAFFOLD --lr 5e-3
python llm_lora_fed.py --dataset pubmed --datapercent 0.005 --fed_alg APFL     --lr 5e-3
python llm_lora_fed.py --dataset pubmed --datapercent 0.005 --fed_alg APPLE    --lr 5e-3
python llm_lora_fed.py --dataset pubmed --datapercent 0.005 --fed_alg fedALA   --lr 5e-3

