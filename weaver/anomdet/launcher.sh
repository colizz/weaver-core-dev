#!/bin/bash

GPU=3;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 100 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 200 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 300 $GPU $SEED & wait
done

GPU=4;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 400 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 500 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 600 $GPU $SEED & wait
done

GPU=5;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 700 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 800 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 900 $GPU $SEED & wait
done

GPU=7;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 1000 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 1100 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 1200 $GPU $SEED & wait
done

## new
GPU=3;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 1600 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 2000 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 2400 $GPU $SEED & wait
done

GPU=4;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 2800 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 3200 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 3600 $GPU $SEED & wait
done

GPU=5;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 4000 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 4400 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 4800 $GPU $SEED & wait
done

GPU=7;
for SEED in 0 1 2 3 4 5; do
./run_dijet_SALAD.sh Wkk 5200 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 5600 $GPU $SEED & ./run_dijet_SALAD.sh Wkk 6000 $GPU $SEED & wait
done


./run_fullspec_SALAD.sh fullspec bin120to130 3 1

$SEED=0
for MASSRANGE in bin70to80 bin80to90 bin90to100 bin100to110 bin110to120 bin120to130; do
./run_fullspec_SALAD.sh fullspec $MASSRANGE $GPU $SEED
done

$SEED=0
for MASSRANGE in bin130to140 bin140to150 bin150to160 bin160to170 bin170to180 bin180to190; do
./run_fullspec_SALAD.sh fullspec $MASSRANGE $GPU $SEED
done

$SEED=0
for MASSRANGE in bin190to200 bin200to210 bin210to220 bin220to230 bin230to240 bin240to250; do
./run_fullspec_SALAD.sh fullspec $MASSRANGE $GPU $SEED
done

$SEED=0
for MASSRANGE in bin240to250 bin250to260 bin260to270 bin270to280 bin280to290 bin290to300; do
./run_fullspec_SALAD.sh fullspec $MASSRANGE $GPU $SEED
done

## formal routine for Wkk

## new
GPU=2;
for SEED in `seq 1 20`; do
./run_dijet_IAD.sh Wkk 400 $GPU $SEED & ./run_dijet_IAD.sh Wkk 800 $GPU $SEED & ./run_dijet_IAD.sh Wkk 1200 $GPU $SEED & wait
done

GPU=3;
for SEED in `seq 1 20`; do
./run_dijet_IAD.sh Wkk 1600 $GPU $SEED & ./run_dijet_IAD.sh Wkk 2000 $GPU $SEED & ./run_dijet_IAD.sh Wkk 2400 $GPU $SEED & wait
done

GPU=4;
for SEED in `seq 1 20`; do
./run_dijet_IAD.sh Wkk 2800 $GPU $SEED & ./run_dijet_IAD.sh Wkk 3200 $GPU $SEED & ./run_dijet_IAD.sh Wkk 3600 $GPU $SEED & wait
done

GPU=5;
for SEED in `seq 1 20`; do
./run_dijet_IAD.sh Wkk 4000 $GPU $SEED & ./run_dijet_IAD.sh Wkk 4400 $GPU $SEED & ./run_dijet_IAD.sh Wkk 4800 $GPU $SEED & wait
done

GPU=6;
for SEED in `seq 1 20`; do
./run_dijet_IAD.sh Wkk 5200 $GPU $SEED & ./run_dijet_IAD.sh Wkk 5600 $GPU $SEED & ./run_dijet_IAD.sh Wkk 6000 $GPU $SEED & wait
done

GPU=7;
for SEED in `seq 1 20`; do
./run_dijet_IAD.sh Wkk 6400 $GPU $SEED & ./run_dijet_IAD.sh Wkk 6800 $GPU $SEED & ./run_dijet_IAD.sh Wkk 7200 $GPU $SEED & wait
done
