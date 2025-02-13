export PYTHONPATH="."
for seed in 830 231 42 9876 2
do
	for entropy_level in "degenerate" "05" "10" "15" "20" "25" "uniform"
	do
		for size in high
		do
			sbatch submit.slurm python systematicity_entropy/eval_rnn.py --model_path systematicity_entropy/trained_models/${size}_${entropy_level}_${seed}_h3/final_model.pt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --task scan --hidden_size 64 --layers 2 --type RNN --wandb ood_exp1_rnn_eval_h3 --entropy_level $entropy_level --size $size --seed $seed
			sbatch submit.slurm python systematicity_entropy/eval_rnn.py --model_path systematicity_entropy/trained_models/${size}_${entropy_level}_${seed}_h2/final_model.pt --val_path data/eight_verbs/full_test/test_full_x_and__7_x.txt --task scan --hidden_size 128 --layers 2 --type RNN --wandb ood_exp1_rnn_eval_h2 --entropy_level $entropy_level --size $size --seed $seed
		done
	done
done
