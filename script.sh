#!/bin/sh

mkdir resultsAlexNet10k_1e-5_64_0.001
mkdir resultsAlexNet10k_1e-5_64_0.0005
mkdir resultsAlexNet10k_1e-5_64_0.00025
mkdir resultsAlexNet10k_1e-5_64_0.0001

python3 main_ddqn.py -n 10000 -e 1e-5 -b 64 -m 10000 -lr 0.001 > resultsAlexNet10k_1e-5_64_0.001/logTraining10k.dat
python3 bestPass.py > resultsAlexNet10k_1e-5_64_0.001/logTest10k.dat
mv models/image* resultsAlexNet10k_1e-5_64_0.001/
mv plots/* resultsAlexNet10k_1e-5_64_0.001/
mv plots_custom/* resultsAlexNet10k_1e-5_64_0.001/
mv results.csv resultsAlexNet10k_1e-5_64_0.001/
mv learning_results.csv resultsAlexNet10k_1e-5_64_0.001/

python3 main_ddqn.py -n 10000 -e 1e-5 -b 64 -m 10000 -lr 0.0005 > resultsAlexNet10k_1e-5_64_0.0005/logTraining10k.dat
python3 bestPass.py > resultsAlexNet10k_1e-5_64_0.0005/logTest10k.dat
mv models/image* resultsAlexNet10k_1e-5_64_0.0005/
mv plots/* resultsAlexNet10k_1e-5_64_0.0005/
mv plots_custom/* resultsAlexNet10k_1e-5_64_0.0005/
mv results.csv resultsAlexNet10k_1e-5_64_0.0005/
mv learning_results.csv resultsAlexNet10k_1e-5_64_0.0005/

python3 main_ddqn.py -n 10000 -e 1e-5 -b 64 -m 10000 -lr 0.00025 > resultsAlexNet10k_1e-5_64_0.00025/logTraining10k.dat
python3 bestPass.py > resultsAlexNet10k_1e-5_64_0.00025/logTest10k.dat
mv models/image* resultsAlexNet10k_1e-5_64_0.00025/
mv plots/* resultsAlexNet10k_1e-5_64_0.00025/
mv plots_custom/* resultsAlexNet10k_1e-5_64_0.00025/
mv results.csv resultsAlexNet10k_1e-5_64_0.00025/
mv learning_results.csv resultsAlexNet10k_1e-5_64_0.00025/

python3 main_ddqn.py -n 10000 -e 1e-5 -b 64 -m 10000 -lr 0.0001 > resultsAlexNet10k_1e-5_64_0.0001/logTraining10k.dat
python3 bestPass.py > resultsAlexNet10k_1e-5_64_0.0001/logTest10k.dat
mv models/image* resultsAlexNet10k_1e-5_64_0.0001/
mv plots/* resultsAlexNet10k_1e-5_64_0.0001/
mv plots_custom/* resultsAlexNet10k_1e-5_64_0.0001/
mv results.csv resultsAlexNet10k_1e-5_64_0.0001/
mv learning_results.csv resultsAlexNet10k_1e-5_64_0.0001/

