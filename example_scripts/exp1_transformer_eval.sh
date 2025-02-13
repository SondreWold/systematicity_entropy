export PYTHONPATH="."
size=high
for seed in 830 231 42 9876 2
do
	for entropy_level in "degenerate" "05" "10" "15" "20" "25" "uniform"
	do
		#sbatch submit.slurm python systematicity_entropy/eval.py --model_path systematicity_entropy/trained_models/${size}_${entropy_level}_${seed}_h2_rel/final_model.pt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --task scan --hidden_size 128 --layers 3 --type rel --wandb ood_exp1_transformer_eval_h2_rel --entropy_level $entropy_level --size $size --seed $seed --input_bucket_size 64 --output_bucket_size 4
		sbatch submit.slurm python systematicity_entropy/eval.py --model_path systematicity_entropy/trained_models/${size}_${entropy_level}_${seed}_h2_rope/final_model.pt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --task scan --hidden_size 128 --layers 3 --type rope --wandb ood_exp1_transformer_eval_h2_rope --entropy_level $entropy_level --size $size --seed $seed --rope_theta 1000
	done
done
