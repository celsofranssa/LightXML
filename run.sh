#DATA=Amazon-670k
#FOLD=0
#
#for i in 0 1 2
#    do
#      python src/cluster.py --dataset $DATA --fold $FOLD --id $i
#	    python src/main.py --lr 1e-4 --epoch 15 --dataset $DATA --fold $FOLD --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i
#	    python src/main.py --lr 1e-4 --epoch 15 --dataset $DATA --fold $FOLD --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i --eval_model
#    done
#    #python src/ensemble_direct.py --model1 Amazon-670k_t0  --model2 Amazon-670k_t1 --model3 Amazon-670k_t2 --dataset $DATA


source ~/projects/venvs/LightXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/LightXML/

train_start=$(date)

DATA=Eurlex-4k
FOLD=0

python src/main.py --lr 1e-4 --epoch 20 --dataset $DATA --fold $FOLD --swa --swa_warmup 10 --swa_step 200 --batch 16
python src/main.py --lr 1e-4 --epoch 20 --dataset $DATA --fold $FOLD --swa --swa_warmup 10 --swa_step 200 --batch 16  --bert roberta
python src/main.py --lr 1e-4 --epoch 20 --dataset $DATA --fold $FOLD --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert xlnet

train_end=$(date)

predict_start=$(date)
python src/ensemble.py --dataset $DATA
predict_end=$(date)



echo "Training Started at $train_start and ended at $train_end"
echo "Prediction Started at $predict_start and ended at $predict_end"


