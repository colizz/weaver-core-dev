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

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom --run-mode train-only --seed $SEED \
--extra-selection "((event_class <= 18) | ((event_class == 21) & ($(get_selection_str)) ))" \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--optimizer ranger ${ext_opt} \
--gpus $GPU --fetch-step 1 --in-memory \
--data-train $DATAFILE_TRAIN \
--data-val $DATAFILE_EVAL \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/anomdet/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

}

# 1. Run IAD
PREFIX=Wkk_IAD.ensem20.64-64.flat.nevt${NEVT}.seed${SEED} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';
# PREFIX=Wkk_IAD.ensem20.512loadffn-64-64.flat.nevt${NEVT}.seed${SEED} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --load-model-weights finetune_pheno.ensemble.0 --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml;

DATAFILE_TRAIN="$DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part0.root";
DATAFILE_EVAL="$DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root";

run_weaver;

# 2. Run SALAD step1

PREFIX=Wkk_SALAD_step1.ensem20.64-64.flat.nevt${NEVT}.seed${SEED} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';
wgtloss_opt=' -o num_ensemble_weight_model 20 -o ft_layer_params_weight_model [(64,0.0),(64,0.0)]'; wgtloss_path=./model/$PREFIX/net_best_epoch_state.pt;
# PREFIX=Wkk_SALAD_step1.ensem20.512loadffn-64-64.flat.nevt${NEVT}.seed${SEED} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --load-model-weights finetune_pheno.ensemble.0 --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';
# wgtloss_opt=' -o num_ensemble_weight_model 20 -o ft_layer_params_weight_model [(512,0.0),(64,0.0),(64,0.0)]'; wgtloss_path=./model/$PREFIX/net_best_epoch_state.pt;

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml;

DATAFILE_TRAIN="$DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part0.root $DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_20ifb.root";
DATAFILE_EVAL="$DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root $DATA_BASEDIR/sm_dijet/mixed_dijet_passsel_ntuple_merged_20ifb.root";

run_weaver;

# 3. Run SALAD step2
# update PREFIX to step2 version

PREFIX=${PREFIX/step1/step2} network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 20 -o use_mlp_shared True -o ft_layer_params [(64,0.0),(64,0.0)] -o merge_after_nth_layer 0  --load-model-weights wgtloss:'${wgtloss_path}' --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';
# PREFIX=${PREFIX/step1/step2} network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 20 -o use_mlp_shared True -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] -o merge_after_nth_layer 0  --load-model-weights finetune_pheno.ensemble.0+wgtloss:'${wgtloss_path}' --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml;

run_weaver;
