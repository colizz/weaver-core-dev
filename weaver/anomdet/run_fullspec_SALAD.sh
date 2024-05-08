#!/bin/bash

SIGNAME=$1
MASSRANGE=$2
GPU=$3
SEED=$4

if [ $SIGNAME != "fullspec" ]; then
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


get_dataloader_config() {

    if [[ $MASSRANGE =~ bin([0-9]+)to([0-9]+) ]]; then
        first_int=${BASH_REMATCH[1]}
        if (( first_int < 180 )); then
            echo -n "--fetch-step 0.1"
        else
            echo -n "--fetch-step 1 --in-memory"
        fi
    fi
}

run_weaver() {

# add seed; use train-only to speed up (optionally)
python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom --run-mode train-only --seed $SEED \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--optimizer ranger ${ext_opt}  \
--gpus $GPU $(get_dataloader_config) \
--data-train $DATAFILE_TRAIN \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/anomdet/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

}

# 2. Run SALAD step1

# PREFIX=fullspec_SALAD_step1_$MASSRANGE.ensem100.64-64.flat.seed${SEED} network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(64,0.0),(64,0.0)] --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 1000000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_fullspec_AD_ensemble.py';
wgtloss_opt=' -o num_ensemble_weight_model 100 -o ft_layer_params_weight_model [(64,0.0),(64,0.0)]'; wgtloss_path=./model/$PREFIX/net_best_epoch_state.pt;
PREFIX=fullspec_SALAD_step1_$MASSRANGE.ensem100.512loadffn-64.flat.seed${SEED} network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.0 --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 1000000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_fullspec_AD_ensemble.py';
wgtloss_opt=' -o num_ensemble_weight_model 100 -o ft_layer_params_weight_model [(512,0.0),(64,0.0)]'; wgtloss_path=./model/$PREFIX/net_best_epoch_state.pt;

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_fullspec/${PREFIX%%.*}.yaml";

DATAFILE_TRAIN=$DATA_BASEDIR'/mixed_ntuple/ntuples_*.root';

run_weaver;


# 3. Run SALAD step2
# update PREFIX to step2 version

# PREFIX=${PREFIX/step1/step2} network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 100 -o ft_layer_params [(64,0.0),(64,0.0)] --load-model-weights wgtloss:'${wgtloss_path}' --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 1000000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_fullspec_AD_ensemble.py';
PREFIX=${PREFIX/step1/step2} network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.0+wgtloss:'${wgtloss_path}' --start-lr 1e-3 --num-epochs 1 --lr-scheduler none --samples-per-epoch 5000000 --samples-per-epoch-val 1000000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_fullspec_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_fullspec/${PREFIX%%.*}.yaml";

run_weaver;
