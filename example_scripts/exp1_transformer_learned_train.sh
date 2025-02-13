export PYTHONPATH="."
for seed in 830 231 42 9876 2
do
	for entropy_level in "degenerate" "05" "10" "15" "20" "25" "uniform"
	do
		sbatch submit.slurm python systematicity_entropy/train.py --train_path data/eight_verbs/datasets/high_${entropy_level}.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path high_${entropy_level}_${seed}_h2_learned --batch_size 32 --epochs 24 --learning_rate 3e-4 --layers 3 --hidden_size 128 --task scan --patience 999 --dropout 0.1 --seed $seed --type learn --rope_theta 1000 --wandb ood_exp1_transformer_h2_learn
	done
done
