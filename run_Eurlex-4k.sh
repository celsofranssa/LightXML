data=Eurlex-4k
fold_idx=5

# Fitting
time_start=$(date '+%Y-%m-%d %H:%M:%S')
python3 src/main.py --lr 1e-4 --epoch 20 --dataset $data --fold $fold_idx --swa --swa_warmup 10 --swa_step 200 --batch 16
python3 src/main.py --lr 1e-4 --epoch 20 --dataset $data --fold $fold_idx --swa --swa_warmup 10 --swa_step 200 --batch 16 --bert roberta
python3 src/main.py --lr 1e-4 --epoch 20 --dataset $data --fold $fold_idx --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert xlnet
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > resource/time/${model}_${data}_fit_${fold_idx}.tmr

# Predicting
time_start=$(date '+%Y-%m-%d %H:%M:%S')
python3 src/ensemble.py --dataset $data
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > resource/time/${model}_${data}_predict_${fold_idx}.tmr

# Evaluating
time_start=$(date '+%Y-%m-%d %H:%M:%S')
python3 eval.py --dataset $data --folds $[fold_idx]
time_end=$(date '+%Y-%m-%d %H:%M:%S')
echo "$time_start,$time_end" > resource/time/${model}_${data}_eval_${fold_idx}.tmr


