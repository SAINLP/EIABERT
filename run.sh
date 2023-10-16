#!/bin/bash
strB='ro'
export epoch=10
export CUDA_VISIBLE_DEVICES=0


data='STS16' #data_name
name='_sim' #'_large', '_sim', '_ro_large', 'ro_sim'
task=$data$name
if [[ $task =~ $strB ]]
then 
	model='roberta-base'
else
	model='bert-base'
fi

for((layer=0;layer<=11;layer++))
do
	python all_run.py \
	--task $task \
	--task_name 'layer' \
	--pretrained_model $model \
	--seed 42 \
	--learning_rate 2e-5 \
	--layer $layer \
	--epoch $epoch
done

python all_run.py \
--task $task \
--task_name '' \
--pretrained_model $model \
--num_layers 3 \
--seed 42 \
--learning_rate 2e-5 \
--epoch $epoch



