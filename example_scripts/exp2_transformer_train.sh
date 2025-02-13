pos_type=van
export PYTHONPATH="."
for seed in 830 231 42 9876 2
do
	for level in 1 2 3 4 5 6 7 8
	do
		epoch=$((64-($level-1)*8))
		# H=3
		sbatch submit.slurm python systematicity_entropy/train.py --train_path data/eight_verbs/full_train/full_${level}_7.txt --val_path data/eight_verbs/full_test/test_full_and_x__x_7.txt --save_path exp2_${level}_${seed}_h3 --batch_size 32 --epochs $epoch --learning_rate 1e-4 --layers 3 --hidden_size 256 --task scan --patience 999 --dropout 0.1 --seed $seed --type $pos_type --wandb ood_exp2_transformer_h3
		# H=2
		sbatch submit.slurm python systematicity_entropy/train.py --train_path data/eight_verbs/full_train/full_${level}_7.txt --val_path data/eight_verbs/full_test/test_full_and_x__x_7.txt --save_path exp2_${level}_${seed}_h2 --batch_size 32 --epochs $epoch --learning_rate 1e-4 --layers 2 --hidden_size 256 --task scan --patience 999 --dropout 0.1 --seed $seed --type $pos_type --wandb ood_exp2_transformer_h2
	done
done
