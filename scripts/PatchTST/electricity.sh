if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchTST

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py
      --random_seed $random_seed
      --is_training 1
      --root_path ./dataset/electricity
      --data_path electricity.csv
      --model_id Electricity96'_'$96
      --model PatchTST
      --data custom
      --features M
      --seq_len 96
      --pred_len 96
      --enc_in 321
      --e_layers 3
      --n_heads 16
      --d_model 128
      --d_ff 256
      --dropout 0.2
      --fc_dropout 0.2
      --head_dropout 0
      --patch_len 16
      --stride 8
      --des 'Exp'
      --train_epochs 100
      --patience 10
      --lradj 'TST'
      --pct_start 0.2
      --itr 1 --batch_size 16 --learning_rate 0.0001 
done
--is_training
1
--root_path
./dataset/electricity
--data_path
electricity.csv
--model_id
electricity1_336'_'96
--model
Time_Unet
--data
ETTh1
--features
M
--seq_len
336
--pred_len
96
--enc_in
321
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
32
--learning_rate
0.005