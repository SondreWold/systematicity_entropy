export PYTHONPATH="."
trans_type=rel
for ibs in 4 8 16 32 64
do
	for obs in 4 8 16 32 64
	do
		sbatch submit.slurm python systematicity_entropy/train.py --train_path data/eight_verbs/datasets/high_20.txt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --save_path sweep_model --batch_size 32 --epochs 24 --learning_rate 3e-4 --layers 3 --hidden_size 128 --input_bucket_size $ibs --output_bucket_size $obs --task scan --patience 999 --dropout 0.1 --seed 42 --type $trans_type --wandb sweep_transformer_relative_exp1
	done
done

