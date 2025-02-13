export PYTHONPATH="."
for seed in 830 231 42 9876 2
do
	for level in 1 2 3 4 5 6 7 8
	do
		sbatch submit.slurm python systematicity_entropy/eval.py --model_path systematicity_entropy/trained_models/exp2_${level}_${seed}_h3/final_model.pt --val_path data/eight_verbs/full_test/test_full_and_x__x_7.txt --task scan --hidden_size 256 --layers 3 --type van --wandb ood_exp2_transformer_eval_h3 --entropy_level $level --seed $seed
		sbatch submit.slurm python systematicity_entropy/eval.py --model_path systematicity_entropy/trained_models/exp2_${level}_${seed}_h2/final_model.pt --val_path data/eight_verbs/full_test/test_full_and_x__x_7.txt --task scan --hidden_size 256 --layers 2 --type van --wandb ood_exp2_transformer_eval_h2 --entropy_level $level --seed $seed
	done
done
