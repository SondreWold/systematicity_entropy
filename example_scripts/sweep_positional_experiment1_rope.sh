export PYTHONPATH="."
trans_type=rope
for rt in 1 10 100 1000 10000 100000 1000000
do
	sbatch submit.slurm python systematicity_entropy/train.py --train_path data/eight_verbs/datasets/high_uniform.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path sweep_model --batch_size 32 --epochs 24 --learning_rate 3e-4 --layers 3 --hidden_size 128 --task scan --patience 999 --dropout 0.1 --seed 42 --type $trans_type --wandb sweep_transformer_rope_exp1 --rope_theta $rt
done

