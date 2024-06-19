#!/bin/bash

SIGNAME=$1
NEVT=$2
GPU=$3
SEED=$4

if [ $SIGNAME != "Wkk" ]; then
    exit
fi
if [ -z $SEED ]; then
    exit
fi
# if zeus is in hostname
if [[ $(hostname) == zeus* ]]; then
    DATA_BASEDIR=/mldata/licq/datasets/JetClassII
elif [[ $(hostname) == farm221* ]]; then
    DATA_BASEDIR=/data/licq/samples/JetClassII
fi

get_selection_str() {
    local step=$((NEVT / 50))
    local start_ind=$((step * SEED))
    local end_ind=$((step * (SEED + 1)))

    local start_mod=$((start_ind / 1000))
    local end_mod=$((end_ind / 1000))
    local start_ind_mod=$((start_ind % 1000))
    local end_ind_mod=$((end_ind % 1000))

    if [[ $start_mod -eq $end_mod ]]; then
        echo -n "(event_no % 1000 >= $start_ind_mod) & (event_no % 1000 < $end_ind_mod)"
    elif [[ $end_mod -eq $((start_mod + 1)) ]]; then
        echo -n "(event_no % 1000 >= $start_ind_mod) | (event_no % 1000 < $end_ind_mod)"
    fi
}

run_weaver() {

# add seed and selection; --run-mode train-only 

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only --seed $SEED \
--extra-selection "((event_class <= 18) | ((event_class == 21) & ($(get_selection_str)) ))" \
--use-amp --optimizer-option weight_decay 0.01 \
--optimizer ranger ${ext_opt} \
--gpus $GPU --fetch-step 0.01 \
--data-train $DATAFILE_TRAIN \
--data-val $DATAFILE_EVAL \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/anomdet/model/formal/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/anomdet/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

}

# 1. Run IAD
PREFIX=Wkk_IAD_lolv.nevt${NEVT}.seed${SEED} network_name=pheno/example_ParticleTransformer2023Dijet; ext_opt='--start-lr 1e-3 --num-epochs 50 --lr-scheduler none --samples-per-epoch 100000 --samples-per-epoch-val 100000 --batch-size 200 --train-mode-params metric:loss --use-last-model';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml;

DATAFILE_TRAIN="$DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part0.root";
DATAFILE_EVAL="$DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root";

run_weaver;
