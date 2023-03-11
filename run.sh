source ~/projects/venvs/LightXML/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/LightXML/

DATA=Eurlex-4k
FOLD=0

python src/main.py --lr 1e-4 --epoch 20 --dataset $DATA --fold $FOLD --swa --swa_warmup 10 --swa_step 200 --batch 16
python src/main.py --lr 1e-4 --epoch 20 --dataset $DATA --fold $FOLD --swa --swa_warmup 10 --swa_step 200 --batch 16  --bert roberta
python src/main.py --lr 1e-4 --epoch 20 --dataset $DATA --fold $FOLD --swa --swa_warmup 10 --swa_step 400 --batch 8 --update_count 2 --bert xlnet