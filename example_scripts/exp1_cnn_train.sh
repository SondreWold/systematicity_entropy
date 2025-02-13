export PYTHONPATH="."
for seed in 830 231 42 9876 2
do
	for entropy_level in "degenerate" "05" "10" "15" "20" "25" "uniform"
	do
		sbatch submit.slurm python systematicity_entropy/train_cnn.py --train_path data/eight_verbs/datasets/low_${entropy_level}.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path low_${entropy_level}_${seed}_cnn --batch_size 32 --epochs 48 --learning_rate 1e-3 --layers 3 --hidden_size 64 --task scan --patience 999 --dropout 0.1 --seed $seed --wandb ood_exp1_cnn --kernel_size 5
		sbatch submit.slurm python systematicity_entropy/train_cnn.py --train_path data/eight_verbs/datasets/medium_${entropy_level}.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path medium_${entropy_level}_${seed}_cnn --batch_size 32 --epochs 36 --learning_rate 1e-3 --layers 3 --hidden_size 64 --task scan --patience 999 --dropout 0.1 --seed $seed --wandb ood_exp1_cnn --kernel_size 5
		sbatch submit.slurm python systematicity_entropy/train_cnn.py --train_path data/eight_verbs/datasets/high_${entropy_level}.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path high_${entropy_level}_${seed}_cnn --batch_size 32 --epochs 24 --learning_rate 1e-3 --layers 3 --hidden_size 64 --task scan --patience 999 --dropout 0.1 --seed $seed --wandb ood_exp1_cnn --kernel_size 5
	done
done
