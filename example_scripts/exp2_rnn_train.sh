export PYTHONPATH="."
for seed in 830 231 42 9876 2
do
	for level in 1 2 3 4 5 6 7 8
	do
		epoch=$((80-($level-1)*8))
		sbatch submit.slurm python systematicity_entropy/train_rnn.py --train_path data/eight_verbs/full_train/full_${level}_7.txt --val_path data/eight_verbs/full_test/test_full_and_x__x_7.txt --save_path exp2_${level}_${seed}_rnn_h3 --batch_size 32 --epochs $epoch --learning_rate 1e-4 --layers 2 --hidden_size 64 --task scan --patience 20 --dropout 0.1 --seed $seed --type GRU --wandb ood_exp2_rnn_h3
		sbatch submit.slurm python systematicity_entropy/train_rnn.py --train_path data/eight_verbs/full_train/full_${level}_7.txt --val_path data/eight_verbs/full_test/test_full_and_x__x_7.txt --save_path exp2_${level}_${seed}_rnn_h2 --batch_size 32 --epochs $epoch --learning_rate 1e-4 --layers 1 --hidden_size 64 --task scan --patience 20 --dropout 0.1 --seed $seed --type RNN --wandb ood_exp2_rnn_h2
	done
done
