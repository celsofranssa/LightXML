
DATA=AmazonCat-13k
FOLD=0

train_start=$(date)
python src/main.py --lr 1e-4 --epoch 5 --dataset $DATA --fold $FOLD --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000
python src/main.py --lr 1e-4 --epoch 5 --dataset $DATA --fold $FOLD --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000 --bert roberta
python src/main.py --lr 1e-4 --epoch 5 --dataset $DATA --fold $FOLD --swa --swa_warmup 2 --swa_step 10000 --batch 32 --eval_step 20000 --bert xlnet --max_len 128
train_end=$(date)

predict_start=$(date)
python src/ensemble.py --dataset $DATA
predict_end=$(date)



echo "Training Started at $train_start and ended at $train_end"
echo "Prediction Started at $predict_start and ended at $predict_end"


