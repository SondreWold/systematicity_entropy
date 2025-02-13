export PYTHONPATH="."
trans_type=van
for lr in 1e-3 3e-3 1e-4 3e-4 1e-5
do
	for hidden_size in 64 128 256
	do
		for dropout in 0.1
		do
			for layers in 1 2 3
			do
					#sbatch submit.slurm python systematicity_entropy/train.py --train_path data/eight_verbs/datasets/high_20.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path sweep_model --batch_size 32 --epochs 30 --learning_rate $lr --layers $layers --hidden_size $hidden_size --task scan --patience 999 --dropout $dropout --seed 42 --type $trans_type --wandb sweep_transformer_exp1
					sbatch submit.slurm python systematicity_entropy/train.py --train_path data/eight_verbs/datasets/high_10.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path sweep_model --batch_size 32 --epochs 30 --learning_rate $lr --layers $layers --hidden_size $hidden_size --task scan --patience 999 --dropout $dropout --seed 42 --type $trans_type --wandb sweep_transformer_exp1
			done
		done
	done
done

