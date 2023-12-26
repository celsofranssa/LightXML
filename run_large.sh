
DATA=Amazon-670k
FOLD=0

train_start=$(date)
for i in 0 1 2
    do
      python3 src/cluster.py --dataset $DATA --fold $FOLD --id $i
	    python3 src/main.py --lr 1e-4 --epoch 15 --dataset $DATA --fold $FOLD --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i
	    python3 src/main.py --lr 1e-4 --epoch 15 --dataset $DATA --fold $FOLD --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i --eval_model
    done
train_end=$(date)

predict_start=$(date)
python3 src/ensemble_direct.py --model1 Amazon-670k_t0  --model2 Amazon-670k_t1 --model3 Amazon-670k_t2 --dataset $DATA
predict_end=$(date)


echo "Training Started at $train_start and ended at $train_end"
echo "Prediction Started at $predict_start and ended at $predict_end"


