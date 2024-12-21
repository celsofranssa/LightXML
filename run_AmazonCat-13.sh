data=AmazonCat-13k
fold_idx=5

# Fitting
time_start=$(date '+%Y-%m-%d %H:%M:%S')
python3 src/main.py --lr 1e-4 --epoch 5 --dataset $data --fold $fold_idx --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000
python3 src/main.py --lr 1e-4 --epoch 5 --dataset $data --fold $fold_idx --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 20000 --bert roberta
python3 src/main.py --lr 1e-4 --epoch 5 --dataset $data --fold $fold_idx --swa --swa_warmup 2 --swa_step 10000 --batch 32 --eval_step 20000 --bert xlnet --max_len 128
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > resource/time/${model}_${data}_fit_${fold_idx}.tmr

# Predicting
time_start=$(date '+%Y-%m-%d %H:%M:%S')
python3 src/ensemble.py --dataset $data
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > resource/time/${model}_${data}_predict_${fold_idx}.tmr

# Evaluating
time_start=$(date '+%Y-%m-%d %H:%M:%S')
python3 eval.py --dataset $data --model $model --fold_idx $fold_idx
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > resource/time/${model}_${data}_eval_${fold_idx}.tmr


