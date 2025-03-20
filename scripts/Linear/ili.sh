# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Time_Unet
for pred_len in 48
do
seq_len=36
while [ $seq_len -le 124 ]
do
python -u run_longExp.py \
zai
let seq_len+=12
done
done

