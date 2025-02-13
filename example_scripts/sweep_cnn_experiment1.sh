export PYTHONPATH="."
for lr in 1e-2 1e-3 1e-4 3e-4
do
	for kernel in 3 5
	do
		for hidden_size in 64 128 256
		do
			for layers in 1 2 3
			do
				for dropout in 0.1
				do
					#sbatch submit.slurm python systematicity_entropy/train_cnn.py --train_path data/eight_verbs/datasets/high_20.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path sweep --batch_size 32 --epochs 30 --learning_rate $lr --layers $layers --hidden_size $hidden_size --task scan --patience 900 --dropout $dropout --seed 42 --wandb sweep_cnn_exp1 --kernel_size $kernel
					sbatch submit.slurm python systematicity_entropy/train_cnn.py --train_path data/eight_verbs/datasets/high_10.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path sweep --batch_size 32 --epochs 30 --learning_rate $lr --layers $layers --hidden_size $hidden_size --task scan --patience 900 --dropout $dropout --seed 42 --wandb sweep_cnn_exp1 --kernel_size $kernel
				done
			done
		done
	done
done
