# LNMLbased_CPD
Code for paper titled "Luckiness Normalized Maximum Likelihood-based Change Detection for High-dimensional Graphical Models with Missing Data".



- Datasets

All of the datasets we mentioned in the paper are in `data/`.

In `synthetic_data/`, there are complete datasets in `complete/`, and corresponding datasets with missing values in `s01/` (10% missing values) and `s025/` (25% missing values). For the datasets with missing values, files ending with `_miss_1` means their missingness is at random, while files ending with `_miss_2` means their missingness is in stripes.

Folder `realworld_data/` includes datasets S&P 500 in `sp500/` and COVID-19 in `covid/`. `sp500/2007_2009_p200_12miss.csv` and `covid/2020_2021_covid.csv` are preprocessed datasets we used in our experiments. And the original datasets are in zipped files.


- Scripts

Please install Python package [pyglassobind](https://github.com/koheimiya/pyglassobind) before running the scripts. 
Script `cpd_LNML.py` outputs the value of MDL change statistics, while `cpd_baseline.py` outputs the gains of baseline methods. The scripts use `s01_p30_*.csv` as an example. All of the Python scripts can be easily run by
```
python *.py
```


- Evaluation

The folder `data/synthetic_data/gain/` includes the MDL change statistics and gains of baseline methods obtained from the synthetic data.
We provide two example scripts to run the evaluation in `gain/s01/p30/`. Both `eval_LNML.py` and `eval_baseline.py` use benefit versus FAR as the criterion.
