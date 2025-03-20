model_name=Time_Unet

for pred_len in 96 192 336 720
do
seq_len=96

python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ETT \
    --data_path ETTm1.csv \
    --model_id ETTm1_$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 336 \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --stage_num 3 \
    --stage_pool_kernel 3 \
    --stage_pool_padding 0 \
    --itr 1 --batch_size 1024 --learning_rate 0.01
done
--is_training
1
--root_path
./dataset/ETT
--data_path
ETTm1.csv
--model_id
ETTm1_336'_'24
--model
Time_Unet
--data
ETTm1
--features
M
--seq_len
336
--pred_len
24
--enc_in
7
--des
'Exp'
--stage_num
3
--stage_pool_kernel
3
--stage_pool_padding
0
--itr
1
--batch_size
1024
--learning_rate
0.01