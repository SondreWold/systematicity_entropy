data_folder=test_folder
for entropy_level in none low medium high higher super uniform
do
	python systematicity_entropy/make_scan_data.py --n 14000 --grammar grammars/train_${entropy_level}.cfg --save_path ${data_folder}/${entropy_level}.txt
done
