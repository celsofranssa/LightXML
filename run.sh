source ~/projects/venvs/LightXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/LightXML/
start=$(date)
DATA=Amazon-670k
FOLD=0

for i in 0 1 2
    do
      python src/cluster.py --dataset $DATA --fold $FOLD --id $i
	    python src/main.py --lr 1e-4 --epoch 15 --dataset $DATA --fold $FOLD --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i
	    python src/main.py --lr 1e-4 --epoch 15 --dataset $DATA --fold $FOLD --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i --eval_model
    done
    #python src/ensemble_direct.py --model1 Amazon-670k_t0  --model2 Amazon-670k_t1 --model3 Amazon-670k_t2 --dataset $DATA

echo "Started at $start and ended at $(date)"


