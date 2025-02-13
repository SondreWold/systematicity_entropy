export PYTHONPATH="."
level=4
epoch=40
for lr in 1e-2 1e-3 3e-3 1e-4 3e-4
do
	for rnn_type in RNN GRU
	do
		for hidden_size in 64 128 256
		do
			for layers in 1 2 3
			do
				for dropout in 0.1
				do
					sbatch submit.slurm python systematicity_entropy/train_rnn.py --train_path data/eight_verbs/full_train/full_${level}_7.txt --val_path data/eight_verbs/full_test/test_full_and_x__x_7.txt --save_path sweep --batch_size 32 --epochs $epoch --learning_rate $lr --layers $layers --hidden_size $hidden_size --task scan --patience 900 --dropout $dropout --seed 42 --wandb sweep_rnn_exp2 --type $rnn_type
				done
			done
		done
	done
done
