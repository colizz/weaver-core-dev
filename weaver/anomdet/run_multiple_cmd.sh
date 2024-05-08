#!/bin/bash

run_job() {

SIGNAME=$1
NEVT=$2
GPU=$3
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p7part.flat.cwgt1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.7)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
# try 5:1? 0.2:1?
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p7part.flat.cwgt0p2-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.7)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p0part.flat.cwgt1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0.ensem20.512-2048.dp0p0part.flat.cwgt1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --load-model-weights finetune_pheno.ensemble.0 --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.ensem20.512-2048.dp0p0part.flat.cwgt1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; ## not work well
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0.ensem20.512-128.flat.cwgt1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(128,0.0)] --load-model-weights finetune_pheno.ensemble.0 --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; ## smaller dim cause it learns less stable...
# default with bs change
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0.ensem20.512-2048.dp0p0part.flat.cwgt2-1or1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --load-model-weights finetune_pheno.ensemble.0 --start-lr 8e-5 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0.ensem20.512-2048.dp0p0part.flat.cwgt2-1or1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --load-model-weights finetune_pheno.ensemble.0 --start-lr 5e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; # Apr26_14-04-33 this is a benchmark
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.ensem20.512-2048.dp0p0part.flat.cwgt2-1or1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --start-lr 5e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
# PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0.ensem20.512-128.dp0p0part.flat.cwgt2-1or1-1; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(128,0.0)] --load-model-weights finetune_pheno.ensemble.0 --start-lr 5e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

# DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/${SIGNAME}${NEVT}/mixed_qcdlt0p1_sig10k_ntuple_cathode_${SIGNAME}${NEVT}_expandsr_dim128_0.root";
# DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/${SIGNAME}${NEVT}/mixed_qcdlt0p1_sig10k_ntuple_cathode_${SIGNAME}${NEVT}_expandsr_dim128_1.root";


## IAD
# PREFIX=${SIGNAME}_IAD_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.64; network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(64,0.0)] --start-lr 1e-3 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_IAD_expandsr_${NEVT}.loadffn01-lrmult0p01.ensem20.512-188-64 network_name=finetune_stage2/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(188,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.all --optimizer-option lr_mult ("model_ensemble.*ft_mlp\.[01].*",0.01) --start-lr 1e-3 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";

config="$HOME/data/pheno/anomdet/cathode/data/classifier/${PREFIX%%.*}.yaml";
rm -rf $HOME/data/pheno/anomdet/cathode/data/classifier/${PREFIX%%.*}.*auto*.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--optimizer ranger ${ext_opt} \
--gpus $GPU --fetch-step 1 --in-memory \
--data-train $DATAFILE_TRAIN \
--data-val $DATAFILE_EVAL \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/anomdet/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

}

# (trap 'kill 0' SIGINT; run_job xbb 10000 0 & run_job xbb 2000 1 & run_job xbs 10000 2 & run_job xbs 2000 3)
(trap 'kill 0' SIGINT; run_job xbb 4000 2 & run_job xbs 4000 3)
