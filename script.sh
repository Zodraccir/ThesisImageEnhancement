#!/bin/sh

mkdir resultsAlexNet40k_1e-5_64_0.001
mkdir resultsAlexNet40k_1e-5_64_0.0005
mkdir resultsAlexNet40k_1e-5_64_0.00025
mkdir resultsAlexNet40k_1e-5_64_0.0001

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -m 10000 -lr 0.001 > resultsAlexNet40k_1e-5_64_0.001/logTraining40k.dat
python3 bestPass.py > resultsAlexNet40k_1e-5_64_0.001/logTest40k.dat
mv models/image* resultsAlexNet40k_1e-5_64_0.001/
mv plots/* resultsAlexNet40k_1e-5_64_0.001/
mv plots_custom/* resultsAlexNet40k_1e-5_64_0.001/
mv results.csv resultsAlexNet40k_1e-5_64_0.001/
mv learning_results.csv resultsAlexNet40k_1e-5_64_0.001/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -m 10000 -lr 0.0005 > resultsAlexNet40k_1e-5_64_0.0005/logTraining40k.dat
python3 bestPass.py > resultsAlexNet40k_1e-5_64_0.0005/logTest40k.dat
mv models/image* resultsAlexNet40k_1e-5_64_0.0005/
mv plots/* resultsAlexNet40k_1e-5_64_0.0005/
mv plots_custom/* resultsAlexNet40k_1e-5_64_0.0005/
mv results.csv resultsAlexNet40k_1e-5_64_0.0005/
mv learning_results.csv resultsAlexNet40k_1e-5_64_0.0005/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -m 10000 -lr 0.00025 > resultsAlexNet40k_1e-5_64_0.00025/logTraining40k.dat
python3 bestPass.py > resultsAlexNet40k_1e-5_64_0.00025/logTest40k.dat
mv models/image* resultsAlexNet40k_1e-5_64_0.00025/
mv plots/* resultsAlexNet40k_1e-5_64_0.00025/
mv plots_custom/* resultsAlexNet40k_1e-5_64_0.00025/
mv results.csv resultsAlexNet40k_1e-5_64_0.00025/
mv learning_results.csv resultsAlexNet40k_1e-5_64_0.00025/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -m 10000 -lr 0.0001 > resultsAlexNet40k_1e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlexNet40k_1e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlexNet40k_1e-5_64_0.0001/
mv plots/* resultsAlexNet40k_1e-5_64_0.0001/
mv plots_custom/* resultsAlexNet40k_1e-5_64_0.0001/
mv results.csv resultsAlexNet40k_1e-5_64_0.0001/
mv learning_results.csv resultsAlexNet40k_1e-5_64_0.0001/

