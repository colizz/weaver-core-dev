# 24.01.15 the delphes GloParT

## pre-training to get weights

PREFIX=JetClassII_full.test
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/merged'
NGPUS=1

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--gpus 1 --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 512 / $NGPUS)) --samples-per-epoch-val $((10000 * 128)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
PREFIX=JetClassII_full.try1
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/merged'
NGPUS=3

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} 

## fixed merging scheme and training size, also split train & val!
PREFIX=JetClassII_full.try2
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/train_*_merged'
NGPUS=3

CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} 

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--gpus 1 --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 512 / $NGPUS)) --samples-per-epoch-val $((10000 * 128)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net --load-epoch 2


### for test

PREFIX=JetClassII_full.test
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/train_*_merged'
NGPUS=1

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--gpus 2 --data-train $DATAPATH'/*0.root' \
--samples-per-epoch $((10 * 512 / $NGPUS)) --samples-per-epoch-val $((10 * 128)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/TEST/net

## 01.18 use scaled vector, final training
PREFIX=JetClassII_full_scale_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/train_*_merged'
NGPUS=3

CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} --load-epoch 34

//predict
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.008 \
--gpus 2 --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX} --load-epoch 30

## 01.19 let's continue with four GPUs..

PREFIX=JetClassII_full_scale_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/train_*_merged'
NGPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} --load-epoch 53

## try 50 epochs?

PREFIX=JetClassII_full_scale_manual.nepoch50
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/train_*_merged'
NGPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} --load-epoch 34

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 2 --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX} --load-epoch 34


## a bb-tagging network 

PREFIX=JetClassII_full_Hbb_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple_merged/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple_merged/*.root'
NGPUS=4

// get weights
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 4 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 9 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX}

## inference scores
// remember to set --test-range!

PREFIX=JetClassII_full_scale_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple_merged/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple_merged/*.root'
NGPUS=1

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.008 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

//
PREFIX=JetClassII_full_Hbb_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple_merged/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple_merged/*.root'
NGPUS=1

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 1 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## Hbb for split QCD

PREFIX=JetClassII_full_Hbb_splitQCD_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple_merged/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple_merged/*.root'
NGPUS=4

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 9 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 9 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX}

DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple_merged/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple_merged/*.root'
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 1 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## Hbb with ParticleNet

PREFIX=JetClassII_full_Hbb_manual.PNet
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple_merged/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple_merged/*.root'
NGPUS=4


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 9 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleNet.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 2 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 9 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleNet.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX}

DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple_merged/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple_merged/*.root'
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 3 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleNet.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 24.02.01 re-train with corrected puppi tune and AK8 jets

## first re-train

PREFIX=JetClassII_ak8puppi_full_scale
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAPATH='/mldata/licq/datasets/JetClassII/train_*_merged'
NGPUS=4

// pre-training
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 3 --data-train $DATAPATH'/*.root' \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX} --load-epoch 49

// export onnx (export_embed = True)
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o fc_params '[(512,0.1)]' -o export_embed True \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_embed.onnx


// predict file for onnx testing
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test /home/pku/licq/pheno/anomdet/gen/delphes_ana/out.root \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_best_epoch_state.pt \
--predict-output /home/pku/licq/pheno/anomdet/gen/delphes_ana/pred.root

// infer scores (repeated in second training, using nonmerged data)
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## hbb and hbs training

PREFIX=JetClassII_ak8puppi_full_hbbonly
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

// pre-training
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX}

// infer scores (repeated in second training, using nonmerged data)
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## 24.02.21 second re-train with mergeQCD
// not using merged dataset, as it causes imbalance between threads

PREFIX=JetClassII_ak8puppi_full_scale_mergeQCD
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_higgspm2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_higgs4p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

// pre-training
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 3 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX} --load-epoch 49

// infer scores
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


### 24.05.03 only H->2prong and QCD

PREFIX=JetClassII_ak8puppi_full_scale_h2pQCD
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_higgspm2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

// pre-training same as above;
// training with NPU=2
NGPUS=2
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 1e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 1e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 1 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX} --load-epoch 0

### make samples-per-epoch smaller

PREFIX=JetClassII_ak8puppi_full_scale_h2pQCD.smallspe
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_higgspm2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'

NGPUS=2
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 1e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((2500 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} --load-epoch 31


// infer sm (the new rule similar with Hbb and Hbs)
// note about *_sminfer.yaml: (1) for xbb, add necessary jet_probs to all dataset. This is only done once because it takes so much time.

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}_sminfer.yaml ## use the special infer dataset

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[(512,0.1)]' \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test \
'mixed_ntuple_1:/mldata/licq/datasets/JetClassII/mixed_ntuple/ntuples_*[13579].root' \
'mixed_xbbbs_ntuple:/mldata/licq/datasets/JetClassII/mixed_xbbbs_ntuple/ntuples_*.root' \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## 24.02.29 Large ParT! (no improvement... deprecated)

PREFIX=JetClassII_ak8puppi_full_scale.large_fc128-1024
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_higgspm2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_higgs4p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

// pre-training
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(128,0),(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--batch-size 256 --start-lr 1e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp -o fc_params '[(128,0),(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--batch-size 256 --start-lr 1e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--use-amp -o fc_params '[(128,0),(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--batch-size 256 --start-lr 1e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 1 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX} --load-epoch 49

// export onnx
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o fc_params '[(128,0),(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o export_embed True \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_embed.onnx

// infer scores
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp -o fc_params '[(128,0),(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o export_embed True \
--batch-size 256 --start-lr 1e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## 24.03.06 final hbb and hbs training, w/ weight decay

PREFIX=JetClassII_ak8puppi_full_hbbonly.wd0p01
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

// have obtained rwgt factors

// train+val

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((2500 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

// infer

DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

PREFIX=JetClassII_ak8puppi_full_scale_hbbonly.wd0p01
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

## 24.04.06 new hbb and hbs training with more hbb/hbs samples... (deprecated)
// more hbb and hbs samples, each with 200 file
// use NGPUS=2

PREFIX=JetClassII_ak8puppi_full_hbbonly.data200fs
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_hbb_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=2

// pre-training
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--use-amp \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((10000 * 1024 / $NGPUS)) --samples-per-epoch-val $((2500 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}


## 24.04.09 final hbb and hbs training with more hbb/hbs samples + weight decay

PREFIX=JetClassII_ak8puppi_full_hbsonly.data200fs.wd0p01
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_hbs_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

PREFIX=JetClassII_ak8puppi_full_hbbonly.data200fs.wd0p01
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_hbb_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

// train+val

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((2500 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

// infer

DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root' # hbb
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_hbs_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root' # hbs

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// infer sm
// note about *_sminfer.yaml: (1) for xbb, add necessary jet_probs to all dataset. This is only done once because it takes so much time.

PREFIX=JetClassII_ak8puppi_full_hbsonly.data200fs.wd0p01
PREFIX=JetClassII_ak8puppi_full_hbbonly.data200fs.wd0p01
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}_sminfer.yaml ## use the special infer dataset

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 2e-3 --num-epochs 80 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test \
'mixed_qcdlt0p1_sig10k_ntuple_merged_1:/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root' \ ## this can be removed
'mixed_ntuple_1:/mldata/licq/datasets/JetClassII/mixed_ntuple/ntuples_*[13579].root' \
'mixed_xbbbs_ntuple:/mldata/licq/datasets/JetClassII/mixed_xbbbs_ntuple/ntuples_*.root' \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### train ParticleNet

PREFIX=JetClassII_ak8puppi_full_hbsonly.data200fs.pnet
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_hbs_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4

PREFIX=JetClassII_ak8puppi_full_hbbonly.data200fs.pnet
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_hbb_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=4


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--backend nccl --data-train $DATAFILE \
--samples-per-epoch $((2500 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleNet.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode val-only \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 0 --data-train $DATAFILE \
--samples-per-epoch $((2500 * 1024 / $NGPUS)) --samples-per-epoch-val $((1000 * 1024)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleNet.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/val.log --tensorboard _${PREFIX}

// infer
DATAFILEINFER='higgs2p:/mldata/licq/datasets/JetClassII/train_higgs2p_ntuple/*.root qcd:/mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root' # hbb

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test $DATAFILEINFER --test-range 0.8 1 \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleNet.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/test.log \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// infer sm
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}_sminfer.yaml ## use the special infer dataset

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 0 \
--data-test \
'mixed_qcdlt0p1_sig10k_ntuple_merged_1:/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root' \
'mixed_ntuple_1:/mldata/licq/datasets/JetClassII/mixed_ntuple/ntuples_*[13579].root' \
'mixed_xbbbs_ntuple:/mldata/licq/datasets/JetClassII/mixed_xbbbs_ntuple/ntuples_*.root' \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleNet.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### fine-tune Sophon to hbs

PREFIX=JetClassII_ak8puppi_full_hbsonly.data200fs.part-ft.onecycle
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}.yaml
DATAFILE='/mldata/licq/datasets/JetClassII/train_hbs_ntuple/*.root /mldata/licq/datasets/JetClassII/train_qcd_ntuple/*.root'
NGPUS=1

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
-o part_fc_params '[(512,0.1)]' -o ft_layer_params '[(512,0.0),(2048,0.0)]' --load-model-weights finetune_pheno.ensemble.0+part \
--batch-size 1024 --start-lr 5e-4 --num-epochs 10 --lr-scheduler one-cycle --optimizer ranger --fetch-step 0.01 \
--gpus 3 --data-train $DATAFILE \
--samples-per-epoch $((250 * 1024 / $NGPUS)) --samples-per-epoch-val $((250 * 1024)) \
--data-config ${config} --num-workers 24 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/pheno/mlp_ft.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

// infer sm
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/${PREFIX%%.*}_sminfer.yaml ## use the special infer dataset

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
-o part_fc_params '[(512,0.1)]' -o ft_layer_params '[(512,0.0),(2048,0.0)]' --load-model-weights finetune_pheno.ensemble.0+part \
--batch-size 1024 --start-lr 5e-4 --num-epochs 10 --lr-scheduler one-cycle --optimizer ranger --fetch-step 0.01 \
--gpus 3 \
--data-test \
'mixed_qcdlt0p1_sig10k_ntuple_merged_1:/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root' \
'mixed_ntuple_1:/mldata/licq/datasets/JetClassII/mixed_ntuple/ntuples_*[13579].root' \
'mixed_xbbbs_ntuple:/mldata/licq/datasets/JetClassII/mixed_xbbbs_ntuple/ntuples_*.root' \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/pheno/mlp_ft.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

------------------------------------------------------------------------------------
# 24.02.07 Starting AD (wrong!!)

PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft64-64-64.dp0p2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2),(64,0.2)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft64-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.1),(64,0.1),(64,0.1)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft64-64-64.dp0p1.nb500; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.1),(64,0.1),(64,0.1)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft128-128-128.dp0p1.nb1000; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.1),(128,0.1),(128,0.1)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft512-128-128.dp0p1.nb1000; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.1),(128,0.1),(128,0.1)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft512-512-512.dp0p1.nb1000; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.1),(512,0.1),(512,0.1)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft512-512-512.dp0p1.nb1000.fullrun; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.1),(512,0.1),(512,0.1)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss'

// should try large dropout
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft512-128-128.dp0p0.nb1000; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(128,0.0)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft512-128-128.dp0p7.nb1000; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.7),(128,0.7),(128,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft512-128-128.dp0p9.nb1000; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.9),(128,0.9),(128,0.9)] --start-lr 5e-3 --num-epochs 32 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_msdgt130.ft512-128-128.dp0p9.nb1000.dlr5e-2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.9),(128,0.9),(128,0.9)] --start-lr 5e-3 --num-epochs 32 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 5e-2'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_ntuple'
GPU=2

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((1000 * 1024)) --samples-per-epoch-val $((1000 * 1024)) \
--batch-size 1024 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 0.01 \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

## 24.02.16 AD testing with Hbb

PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p7; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p9; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.9),(64,0.9),(64,0.9)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 5e-2'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p95.earlystop; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95),(64,0.95)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p95; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95),(64,0.95)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p95.earlystop.lb; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95),(64,0.95)] --start-lr 5e-1 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p95.lb.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95),(64,0.95)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p9.lb.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.9),(64,0.9),(64,0.9)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss' # using bs=100000
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p9.try3; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.9),(64,0.9),(64,0.9)] --start-lr 2e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p95.try3; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95),(64,0.95)] --start-lr 2e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss' # this is a good one!
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p95.try4; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95),(64,0.95)] --start-lr 1e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft64-64-64.dp0p95.try4; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95),(64,0.95)] --start-lr 1e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss'

// back to small bs
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft128-128-128.dp0p95.es.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.95),(128,0.95),(128,0.95)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop' ## large success!
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft256-256-256.dp0p98.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(256,0.98),(256,0.98),(256,0.98)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbtest.ft32-32-32.dp0p9.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(32,0.9),(32,0.9),(32,0.9)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=2

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((100 * 1000)) --samples-per-epoch-val $((100 * 1000)) \
--batch-size 1000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}



// (prediction only)
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--samples-per-epoch $((20 * 10000)) --samples-per-epoch-val $((20 * 10000)) \
--batch-size 10000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 0.1 \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_epoch-15_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred_epoch-15.root



## 24.02.16 AD with multi-binned Hbb
PREFIX=JetClassII_ak8puppi_AD_hbbmultibin.ft128-128-128.dp0p95.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.95),(128,0.95),(128,0.95)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbmultibin.ft128-128-128.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.7),(128,0.7),(128,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbmultibin.ft128-128-128.dp0p9.es.try3; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.9),(128,0.9),(128,0.9)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_hbbmultibin.ft128-128-128.dp0p8.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.8),(128,0.8),(128,0.8)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=2

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((100 * 1000)) --samples-per-epoch-val $((100 * 1000)) \
--batch-size 1000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


PREFIX=JetClassII_ak8puppi_AD_fullrange.ft128-128-128.dp0p9.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.9),(128,0.9),(128,0.9)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft128-128-128.dp0p9.es.dlr5e-3; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.9),(128,0.9),(128,0.9)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 5e-3'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft128-128-128.dp0p95.es.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.95),(128,0.95),(128,0.95)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft128-128-128.dp0p95.norwgt; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.95),(128,0.95),(128,0.95)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop' // should disable rwgt
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft128-128-128.dp0p95.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.95),(128,0.95),(128,0.95)] --start-lr 1e-2 --num-epochs 20 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8' # switch to bs 10000
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft6-6-6.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(6,0),(6,0),(6,0)] --start-lr 1e-2 --num-epochs 20 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ftnull.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [] --start-lr 1e-2 --num-epochs 20 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft256.dp0p95.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(256,0.95)] --start-lr 4e-3 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8' # switch to bs 10000
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft256.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(256,0.0)] --start-lr 4e-3 --num-epochs 100 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'

// it seems dropout=0 results in better shape..
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft256.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(256,0.0)] --start-lr 2e-3 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft256.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(256,0.0)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft512-256.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(256,0.0)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft512-256-256.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(256,0.0),(256,0.0)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8' // no need to have three layers
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft512-256.dp0p3.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.3),(256,0.3)] --start-lr 2e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft64-64.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft64-64.dp0p1.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.1),(64,0.1)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft64-64.dp0p2.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_fullrange.ft64-64.dp0p3.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.3),(64,0.3)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=3

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 10000)) --samples-per-epoch-val $((10 * 10000)) \
--batch-size 10000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


// official production using one-cycle (1-epoch)
// test results: this AD does not look optimal.. still cannot reproduce cms result!

PREFIX=JetClassII_ak8puppi_AD_fullrange.ft64-64.dp0p2.onecycle.ofc; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-2 --num-epochs 1 --lr-scheduler one-cycle --train-mode-params metric:loss'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=3

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

for i in `seq 1 100`; do
python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--samples-per-epoch $((500 * 10000)) --samples-per-epoch-val $((500 * 10000)) \
--batch-size 10000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_try${i};
done

// predict
for i in `seq 2 100`; do
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--samples-per-epoch $((500 * 10000)) --samples-per-epoch-val $((500 * 10000)) \
--batch-size 10000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-test ${DATAPATH}_merged_1.root \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_try${i}_epoch-0_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred_try${i}.root;
done


## 24.02.18 AD for Xbb/Xbs

PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64-64.dp0p7; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64-64.dp0p7.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 10 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64.dp0p2.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-2 --num-epochs 50 --lr-scheduler one-cycle --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64-64.dp0p9; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.9),(64,0.9),(64,0.9)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ftfromprobs.ft64-64-64.dp0p9; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(188,0,"idx:0,161:188"),(64,0.9),(64,0.9),(64,0.9)] --load-model-weights finetune_pheno.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 5e-2 --num-epochs 16 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ftfromprobs.ft64-64-64.dp0p2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(188,0,"idx:0,161:188"),(64,0.2),(64,0.2),(64,0.2)] --load-model-weights finetune_pheno.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 5e-2 --num-epochs 16 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'

// continue..
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64.dp0p0; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 32 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64.dp0p95; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.95),(64,0.95)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft200-200.dpmax; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(200,0.99),(200,0.99)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft2-2.dp0p0; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(2,0.0),(2,0.0)] --start-lr 5e-2 --num-epochs 16 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft200-200.dpmax.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(200,0.99),(200,0.99)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8' // good!!

// add tensorboard_fn
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft200-200.dpmax.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(200,0.99),(200,0.99)] --start-lr 5e-3 --num-epochs 32 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64.dp0p0.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 32 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft200-200.dpmax.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(200,0.99),(200,0.99)] --start-lr 5e-3 --num-epochs 32 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64-64.dp0p7.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.smdiv4.ft64-64-64.dp0p7.tb_fn.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py' // this seems good
PREFIX=JetClassII_ak8puppi_AD_xbbtest.smdiv4.ft64-64-64.dp0p7.tb_fn.try3; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.smdiv10.ft64-64-64.dp0p7.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'

// trying IAD? (24.02.20)
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.ft64-64.dp0p7; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.ft64-64.dp0p7.es.try2.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py' ## -> a stable good solution for IAD!
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.ft64-64.dp0p7.es.try2.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=1

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 10000)) --samples-per-epoch-val $((10 * 10000)) \
--batch-size 10000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


// (prediction only)
i=16
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--samples-per-epoch $((20 * 10000)) --samples-per-epoch-val $((20 * 10000)) \
--batch-size 10000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 0.1 \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_epoch-${i}_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred_epoch-${i}.root


// shall we switch back to small batch-size?

PREFIX=JetClassII_ak8puppi_AD_xbbtest.smdiv10.ft64-64-64.dp0p7.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest.smdiv4.ft64-64-64.dp0p7.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'

PREFIX=JetClassII_ak8puppi_AD_xbbtest.ft64-64.dp0p2.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 2.5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8 --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'

// try IAD (24.02.20)
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs500.ft64-64.dp0p7.es.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 2.5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs500.ft64-64.dp0p8.es.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.8),(64,0.8)] --start-lr 2.5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=1

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((100 * 500)) --samples-per-epoch-val $((100 * 500)) \
--batch-size 500 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

(very large bins) 
=> conclusion: the larger the more stable!
=> lr should be set equal, if batch-per-epoch does not change
=> you should stop when dlr reaches a plato, not using dlr > threshold..

PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs200000.ft64-64.dp0p7.es.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 2000000 --samples-per-epoch-val 2000000 --batch-size 200000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs50000.ft64-64.dp0p7.es.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py' ## -> a stable good solution for IAD? not really..

PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs50000.ft64-64.dp0p9.es.tb_fn; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.9),(64,0.9)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=1

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


// back to multibins

PREFIX=JetClassII_ak8puppi_AD_fullrange_xbb.bs500.ft64-64.dp0p2.onecycle; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 2e-3 --num-epochs 50 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple'
GPU=1

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((200 * 500)) --samples-per-epoch-val $((200 * 500)) \
--batch-size 500 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

## 24.02.24 IAD, modified

// remember to change to tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py

PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs50000.ftfromprobs.ft64-64.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(162,0,"idx:0,161"),(64,0.7),(64,0.7)] --load-model-weights finetune_pheno_mergeQCD.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs50000.ft64-64.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs500.ft64-64.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 1000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'

PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs500.ft128-128.dp0p9.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.9),(128,0.9)] --start-lr 2e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs500.64-64-64.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 2e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'

DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_2'
GPU=3

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// run repeative routines
for i in `seq 1 6`; do \
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs500.256-256.dp0p9.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(256,0.9),(256,0.9)] --start-lr 2e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py';
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}; done

for i in `seq 1 6`; do \
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs500.64-64-64.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 2e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py';
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}; done

for i in `seq 1 6`; do \
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs500.64-64-64.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 2e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py';
DATAPATH='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple';
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}; done

// is it the optimizer problem?

PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs50000.ft64-64.dp0p7.es.sgd; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7)] --start-lr 2e-1 --num-epochs 64 --lr-scheduler cosanneal --optimizer sgd --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs50000.ft64-64.dp0p0.es.sgd; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0)] --start-lr 2e-1 --num-epochs 64 --lr-scheduler cosanneal --optimizer sgd --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'

# 24.02.25 Second AD attempt
// should not use "--fetch-by-files --fetch-step 1 --in-memory" if you have several files!!!


PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs500.64-64-64.dp0p7.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 2e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.sm_2.bs500.64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_2.py'

DATAFILE_0='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_2_merged_0.root'
DATAFILE_1='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_2_merged_1.root'
GPU=3

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 1 --in-memory \
--data-train $DATAFILE_0 \
--data-val $DATAFILE_1 \
--data-test $DATAFILE_1 \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

## now tries ID, original model

// let's first see (again) what happens if you train it thouroughly
PREFIX=JetClassII_ak8puppi_AD_xbbtest.bs500.128-128.dp0p0; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(128,0.),(128,0.)] --start-lr 2e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'
// or we still set large dropout
PREFIX=JetClassII_ak8puppi_AD_xbbtest.bs500.64-64-64.dp0p8.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.8),(64,0.8),(64,0.8)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py'

// back to IAD. The last check
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs500.64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py' ## this performs fine

PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.64-64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2),(64,0.2)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.512-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.2),(64,0.2)] --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.loadffn.512-188-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(188,0),(64,0.2)] --load-model-weights finetune_pheno.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.loadffnft.512-188-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.2),(188,0.2),(64,0.2)] --load-model-weights finetune_pheno.all --optimizer-option lr_mult ("ft_mlp.[01].*",0.1) --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.loadffnft.512-188-64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.2),(188,0.2),(64,0.2),(64,0.2)] --load-model-weights finetune_pheno.all --optimizer-option lr_mult ("ft_mlp.[01].*",0.1) --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'

// limited nodes?
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.loadffnss.512-188-64.dp0p2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(188,0,"idx:6,161:188"),(64,0.2)] --load-model-weights finetune_pheno.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.sm_2.bs500.loadffnss.512-188-64.dp0p2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(162,0,"idx:6,161"),(64,0.2)] --load-model-weights finetune_pheno_mergeQCD.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.sm_2.bs500.loadffnss.512-188-64.dp0p0.wd0p1; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(162,0,"idx:6,161"),(64,0)] --load-model-weights finetune_pheno_mergeQCD.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --optimizer-option weight_decay 0.1 --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_2.py' ## =>load the sm_2 samples

PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.load1stffn.512.dp0p0; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0)] --load-model-weights finetune_pheno.0 --optimizer-option lr_mult ("ft_mlp.[0].*",0) --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.load1stffn.512-64.dp0p0; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(64,0)] --load-model-weights finetune_pheno.0 --optimizer-option lr_mult ("ft_mlp.[0].*",0) --start-lr 1e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'

// change loss func
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs500.64-64.dp0p2.es.mseloss; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] -o loss_type "cls:mse" --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
// SM div by 5 or 20: indeed help..
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD_smdiv5.bs500.64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 50000 --samples-per-epoch-val 50000 --batch-size 500 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'

// change bs?
// these lessons are real:
// => large bs is better -> denser point to avoid wrong direction due to local stat fluctuation
// => and large network capacity (if input is 2D, the data is much denser to avoid overfitting)...

PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs50000.64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbbtest_IAD.bs50000.64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.2),(64,0.2)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest.py' ## bb
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.sm_2.bs50000.loadffnss.512-188-64.dp0p2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(162,0,"idx:6,161"),(64,0.2)] --load-model-weights finetune_pheno_mergeQCD.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.sm_2.bs50000.loadffnss.512-188-64-64-64.dp0p2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(162,0,"idx:6,161"),(64,0.2),(64,0.2),(64,0.2)] --load-model-weights finetune_pheno_mergeQCD.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_2.py' ## this works!!
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.sm_2.bs50000.loadffn.512-188-64-64-64.dp0p2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0),(162,0),(64,0.2),(64,0.2),(64,0.2)] --load-model-weights finetune_pheno_mergeQCD.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_2.py' # overfitting

PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs50000.512-64-64-64.dp0p2.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.2),(64,0.2),(64,0.2),(64,0.2)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py' # well this tends to be stable!
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs50000.512-128-64-64-64.dp0p0.es.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py' # => this is the best config!

PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs200000.512-128-64-64-64.dp0p0.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 1000000 --samples-per-epoch-val 1000000 --batch-size 200000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'

PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.sm_2.bs50000.512-128-64-64-64.dp0p0.es.try2; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_2.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs50000.64-64-64re-64-64re.dp0p0.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0),(64,0.0,"resid:0"),(64,0.0),(64,0.0,"resid:2")] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.bs50000.64-64-64.dp0p0.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'

DATAFILE_0='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_merged_0.root'
DATAFILE_1='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_merged_1.root'
GPU=1

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 1 --in-memory \
--data-train $DATAFILE_0 \
--data-val $DATAFILE_1 \
--data-test $DATAFILE_1 \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// train on huge dataset

PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.fullmix.bs50000.64-64-64.dp0p0.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_IAD.fullmix.bs200000.64-64-64.dp0p0.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(64,0.0),(64,0.0),(64,0.0)] --start-lr 2e-1 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 1000000 --samples-per-epoch-val 1000000 --batch-size 200000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'


DATAPATH='/mldata/licq/datasets/JetClassII/mixed_ntuple'
GPU=2

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 \
--data-train $DATAPATH'/*[02468].root' \
--data-val $DATAPATH'/*[13579].root' \
--data-test $DATAPATH'/*[13579].root' \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


## re-try fullrange

PREFIX=JetClassII_ak8puppi_AD_fullrange_xbb.bs50000.512-128-64-64-64.dp0p0.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss'

DATAFILE_0='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_merged_0.root'
DATAFILE_1='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_merged_1.root'
GPU=1

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((200 * 500)) --samples-per-epoch-val $((200 * 500)) \
--batch-size 500 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 1 --in-memory \
--data-train $DATAFILE_0 \
--data-val $DATAFILE_1 \
--data-test $DATAFILE_1 \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

# 24.02.28 CATHODE classifier

PREFIX=JetClassII_ak8puppi_AD_xbstest_cathode.bs50000.512-128-64-64-64.dp0p0.es; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'
PREFIX=JetClassII_ak8puppi_AD_xbstest_cathode.bs50000.512-128-64-64-64.dp0p0; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 15 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --tensorboard-custom-fn tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py'

DATAFILE_0='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_merged_0.root /mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_xbs_sr_sampled_0.root'
DATAFILE_1='/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_merged_1.root /mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_ntuple_xbs_sr_sampled_1.root'
GPU=1

config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 1 --in-memory \
--data-train $DATAFILE_0 \
--data-val $DATAFILE_1 \
--data-test $DATAFILE_1 \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


# 24.04.10 CATHODE classifier official
// use the previous configs (15 epochs)
// should cd anomdet/

NEVT=4000; NTRY=1

for NEVT in 4000 1000 2000 3000 6000 8000 10000; do
for NTRY in `seq 1 10`; do
PREFIX=xbs_cathode_classifier_${NEVT}.ofc.try${NTRY}; network_name=pheno/mlp; ext_opt='-o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 15 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --early-stop --early-stop-dlr 1e8  --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs${NEVT}_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs${NEVT}_1.root";
DATAFILE_TEST="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";
GPU=1

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--batch-size 50000 --start-lr 5e-3 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-step 1 --in-memory \
--data-train $DATAFILE_TRAIN \
--data-val $DATAFILE_EVAL \
--data-test $DATAFILE_TEST \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/anomdet/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/an'om'de'tlogs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

done; done

# 24.04.10 CATHODE: train an ensemble at one time

NEVT=10000

PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.ofc.ensemble; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 15 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.512-128-64-64-64.dp0p7; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.7),(128,0.7),(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-2 --num-epochs 15 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.512-128-64-64-64.dp0p0.onecycle; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 15 --lr-scheduler one-cycle --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.512-128-64-64-64.dp0p0.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 2e-3 --num-epochs 50 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py'; # this is the standard one
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.512-512-512-512-512.dp0p0.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(512,0.0),(512,0.0),(512,0.0),(512,0.0)] --start-lr 5e-4 --num-epochs 50 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.2048-128-64-64-64.dp0p0.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(2048,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 2e-3 --num-epochs 50 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs${NEVT}_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs${NEVT}_1.root";
DATAFILE_TEST="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";
GPU=3

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";

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
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};


## for IAD
NEVT=4000
PREFIX=xbs_IAD_ensemble_${NEVT}.512-128-64-64-64.dp0p0.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 2e-3 --num-epochs 50 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_IAD_ensemble_${NEVT}.loadffn.512-188-64-64-64.dp0p0.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(188,0.0),(64,0.0),(64,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.all --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.[01].*",0) --start-lr 2e-3 --num-epochs 50 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';

// change back to cosanneal
NEVT=2000
PREFIX=xbs_IAD_ensemble_${NEVT}.512-128-64-64-64.dp0p0; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(128,0.0),(64,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_IAD_ensemble_${NEVT}.512-128-64-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.1),(128,0.1),(64,0.1),(64,0.1),(64,0.1)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';

// now you understand: interpolation is the most important..

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";
GPU=1

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";

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
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

# 24.04.18 Formal try for finalising the study
NEVT=10000
PREFIX=xbs_IAD_ensemble_${NEVT}.ensem100.512-64-128-64re-128-64re.dp0p0; network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0),(128,0.0),(64,0.0,"resid:1"),(128,0.0),(64,0.0,"resid:3")] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_IAD_ensemble_${NEVT}.ensem100.512-64-64; network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_IAD_ensemble_${NEVT}.ensem50.512-64-64.dp0p9; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.9),(64,0.9),(64,0.9)] --start-lr 2e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
// change bs again to 1024, 5000 or 250000
PREFIX=xbs_IAD_ensemble_${NEVT}.ensem100.512-64-64; network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] --start-lr 5e-4 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 1024 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py'; // this doesn't work even for 12000
PREFIX=xbs_IAD_ensemble_${NEVT}.ensem25.512-64-64; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';

// xbb
PREFIX=xbb_IAD_ensemble_${NEVT}.ensem25.512-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_ensemble.py';
PREFIX=xbb_IAD_hlvars_ensemble_${NEVT}.ensem25.128-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(128,0.1),(64,0.1),(64,0.1)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";


DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";
GPU=2

// on farm221
DATAFILE_TRAIN="/data/licq/samples/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/data/licq/samples/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";
GPU=5


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
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};


## LaCATHODE classifier (doesn't work)
NEVT=10000
PREFIX=xbs_lacathode_classifier_${NEVT}.ensem100.512-64-64; network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_lacathode_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";


DATAFILE_TRAIN="/data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_merged_0_xbs1000_dim128.root /mldata/licq/datasets/JetClassII/gaussian_0.root";
DATAFILE_EVAL="/data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_merged_1_xbs1000_dim128.root /mldata/licq/datasets/JetClassII/gaussian_1.root";
GPU=3

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
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

## back to CATHODE
NEVT=10000
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.ensem50.512-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --start-lr 5e-2 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.ensem50.512-64-64.dp0p9; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.9),(64,0.9),(64,0.9)] --start-lr 2e-2 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';
PREFIX=xbs_cathode_classifier_ensemble_${NEVT}.ensem50.512-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --start-lr 2e-3 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbstest_ensemble.py';

// xbb test
PREFIX=xbb_cathode_classifier_ensemble_${NEVT}.ensem50.512-64-64.dp0p1.test; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --start-lr 2e-3 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_ensemble.py';
PREFIX=xbb_cathode_classifier_ensemble_${NEVT}.ensem50.512-64-64.dp0p1.flat.test; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --start-lr 2e-4 --num-epochs 32 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_xbbtest_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs1000_dim128_try2_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs1000_dim128_try2_1.root";
GPU=3

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";

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
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

# 24.04.21 Expand the SR...
// unify the tb custom function!

NEVT=2000
PREFIX=xbs_IAD_expandsr_ensemble_${NEVT}.ensem25.512-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --start-lr 5e-2 --num-epochs 16 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=xbs_IAD_expandsr_ensemble_${NEVT}.loadffn0.ensem25.512-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 5e-2 --num-epochs 16 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; // wow, this works!
// tuning lrmult=0.1 seems better
NEVT=10000
PREFIX=xbs_IAD_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p1.ensem25.512-64-64.dp0p1; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.1) --start-lr 5e-2 --num-epochs 16 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
// re-test
NEVT=4000
PREFIX=xbs_IAD_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p01.ensem10.512-2048.dp0p2part.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 10 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-3 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=xbb_IAD_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p01.ensem10.512-2048.dp0p2part.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 10 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';


config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";
#GPU=1

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
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

## back to CATHODE

PREFIX=xbs_cathode_classifier_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p1.ensem50.512-64-64.dp0p1.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(64,0.1),(64,0.1)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.1) --start-lr 2e-3 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=xbs_cathode_classifier_expandsr_ensemble_${NEVT}.loadffn-lrmult0p1.ensem50.512-188-64-64.dp0p1.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(188,0.1),(64,0.1),(64,0.1)] --load-model-weights finetune_pheno.ensemble.all --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.[01].*",0.1) --start-lr 2e-3 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; // worse
PREFIX=xbs_cathode_classifier_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p01.ensem50.512-256-64.dp0p1.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 50 -o ft_layer_params [(512,0.1),(256,0.1),(64,0.1)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 2e-3 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; // increasing feat dim!
PREFIX=xbs_cathode_classifier_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p01.ensem25.512-512-512.dp0p1.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 25 -o ft_layer_params [(512,0.1),(512,0.1),(512,0.1)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 2e-3 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=xbs_cathode_classifier_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p01.ensem10.512-1024-512.dp0p1.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 10 -o ft_layer_params [(512,0.1),(1024,0.1),(512,0.1)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 2e-3 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; // change ensemble num
PREFIX=xbs_cathode_classifier_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p01.ensem10.512-2048.dp0p2part.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 10 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 2e-3 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; // dropout to partial layer
NEVT=2000
PREFIX=xbs_cathode_classifier_expandsr_ensemble_${NEVT}.loadffn0-lrmult0p01.ensem10.512-2048.dp0p2part.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 10 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';


DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs2000_expandsr_dim128_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_xbs2000_expandsr_dim128_1.root";

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";

// to formal training
NEVT=10000
SIGNAME=xbb
PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem10.512-2048.dp0p2part.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 10 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p2part.flat; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 8 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_${SIGNAME}${NEVT}_expandsr_dim128_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_${SIGNAME}${NEVT}_expandsr_dim128_1.root";

config="$HOME/data/pheno/anomdet/cathode/data/classifier/${PREFIX%%.*}.yaml";

# 24.04.23 Back to CWoLa
NEVT=10000
SIGNAME=xbb
PREFIX=${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.ensem20.512-2048.dp0p2part; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.ensem20.512-128-128; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(128,0.0),(128,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.ensem20.32-32; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(32,0.0),(32,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.ensem20.null; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.ensem20.32; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(32,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD/cathode/${PREFIX%%.*}.yaml";

// second step, another MLP with weighted sample loss
NEVT=10000
SIGNAME=xbb
wgtloss_opt=' -o num_ensemble_weight_model 20 -o ft_layer_params_weight_model [(512,0.0),(128,0.0),(128,0.0)]'; wgtloss_path=$HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.ensem20.512-128-128/net_best_epoch_state.pt
PREFIX=${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.wgtloss-step2.loadffn0-lrmult0p01.ensem10.512-2048.dp0p2part; network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 10 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0+wgtloss:'${wgtloss_path}' --optimizer-option lr_mult ("m_main.model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

wgtloss_opt=' -o num_ensemble_weight_model 20 -o ft_layer_params_weight_model [(32,0.0)]'; wgtloss_path=$HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.ensem20.32/net_best_epoch_state.pt
PREFIX=${SIGNAME}_cwola_classifier_expandsr_ensemble_${NEVT}.wgtloss-step2.loadffn0-lrmult0p01.ensem10.512-2048.dp0p2part.flat; network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 10 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0+wgtloss:'${wgtloss_path}' --optimizer-option lr_mult ("m_main.model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-3 --num-epochs 64 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";

## check the distribution

NEVT=10000
SIGNAME=xbb
PREFIX=${SIGNAME}_cwola_multibin_${NEVT}.ensem20.512-256-256; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(256,0.0),(256,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_cwola_multibin_${NEVT}.ensem20.512-2048; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
PREFIX=${SIGNAME}_cwola_multibin_full_${NEVT}.ensem15.512-2048; network_name=pheno/mlp; ext_opt='-o num_ensemble 15 -o ft_layer_params [(512,0.0),(2048,0.0)] --start-lr 5e-2 --num-epochs 64 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

# 24.04.25 CATHODE proceed with bidirection

NEVT=10000
SIGNAME=xbs
PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p2part.flat.bidirect; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
// a comparison between two modes
PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p2part.bidirect; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 1e-2 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py'; // bidirectral way has better bkg estimation! But there are few events with high score that hides the signal...

PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p2part.flat.bidirect.cwgt1-1; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.2)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_bidirect_${SIGNAME}${NEVT}_expandsr_dim128_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_bidirect_${SIGNAME}${NEVT}_expandsr_dim128_1.root";

// what about using original cathode, but with class weights >1:1?
NEVT=10000
SIGNAME=xbs
PREFIX=${SIGNAME}_cathode_classifier_expandsr_${NEVT}.loadffn0-lrmult0p01.ensem20.512-2048.dp0p7part.flat.cwgt1-1; network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(2048,0.7)] --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp_list.0.*",0.01) --start-lr 4e-4 --num-epochs 16 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 50000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_${SIGNAME}${NEVT}_expandsr_dim128_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root /data/pku/home/licq/pheno/anomdet/cathode/output/mixed_qcdlt0p1_sig10k_ntuple_cathode_${SIGNAME}${NEVT}_expandsr_dim128_1.root";


config="$HOME/data/pheno/anomdet/cathode/data/classifier/${PREFIX%%.*}.yaml";
rm -rf $HOME/data/pheno/anomdet/cathode/data/classifier/${PREFIX%%.*}.*auto*.yaml

# 24.04.30 new IAD
SIGNAME=xbb; NEVT=4000;

PREFIX=${SIGNAME}_IAD_expandsr_${NEVT}.loadffn01-lrmult0p01.ensem20.512-188-64 network_name=pheno/mlp; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(188,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.all --optimizer-option lr_mult ("model_ensemble.*ft_mlp\.[01].*",0.01) --start-lr 1e-3 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';

config="$HOME/data/pheno/anomdet/cathode/data/classifier/${PREFIX%%.*}.yaml";
rm -rf $HOME/data/pheno/anomdet/cathode/data/classifier/${PREFIX%%.*}.*auto*.yaml

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/mixed_qcdlt0p1_sig10k_ntuple_merged_1.root";
#GPU=1

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
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};



------------------------------------------
# 24.04.06 evt class for leyun

PREFIX=evt_full.test
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/EvtClass/${PREFIX%%.*}.yaml
DATAFILE='/home/olympus/lyazj/software/ad-part/run/sm/data4/*/*.root'

// have obtained rwgt factors

// train+val

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' -o num_heads 4 --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 2 --fetch-step 1 --in-memory \
--samples-per-epoch $((512 * 1500)) --samples-per-epoch-val $((512 * 375)) \
--data-train $DATAFILE --data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023EvtClassifier.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

// only using jet input with hidden neuron
// card: evt_jethid.yaml; specify "-o input_dims '(128,)'" to use only one input group

PREFIX=evt_jethid.test
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/EvtClass/${PREFIX%%.*}.yaml
DATAFILE='/home/olympus/lyazj/software/ad-part/run/sm/data4/*/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o embed_dims '[1024,256,64]' -o pair_embed_dims '[32,32,32]' -o num_heads 4 -o input_dims '(128,)' --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 3 --fetch-step 1 --in-memory \
--samples-per-epoch $((512 * 1500)) --samples-per-epoch-val $((512 * 375)) \
--data-train $DATAFILE --data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023EvtClassifier.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}


PREFIX=evt_full_hid.test
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/EvtClass/${PREFIX%%.*}.yaml
DATAFILE='/home/olympus/lyazj/software/ad-part/run/sm/data4/*/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp -o embed_dims '[1024,256,64]' -o pair_embed_dims '[32,32,32]' -o num_heads 4 --optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 3 --fetch-step 1 --in-memory \
--samples-per-epoch $((512 * 1500)) --samples-per-epoch-val $((512 * 375)) \
--data-train $DATAFILE --data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023EvtClassifier.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}


// training SimpleTransformer for test
PREFIX=evt_full.simTF
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/EvtClass/${PREFIX%%.*}.yaml
DATAFILE='/home/olympus/lyazj/software/ad-part/run/sm/data4/*/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--optimizer-option weight_decay 0.01 \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 \
--gpus 2 --fetch-step 1 --in-memory \
--samples-per-epoch $((512 * 1500)) --samples-per-epoch-val $((512 * 375)) \
--data-train $DATAFILE --data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/Transformer.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

// training SimpleTransformerVAE for test

PREFIX=evt_full.simTransformerVAE.unembed.beta0
PREFIX=evt_full.simTransformerVAE.unembed.beta0.heavyembed
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/EvtClass/${PREFIX%%.*}.yaml
DATAFILE='/home/olympus/lyazj/software/ad-part/run/sm/data4/QCD/0.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom \
--optimizer-option weight_decay 0.01 -o kld_weight 0 \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--gpus 3 --fetch-step 1 --in-memory \
--samples-per-epoch $((512 * 1500)) --samples-per-epoch-val $((512 * 375)) \
--data-train $DATAFILE --data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/TransformerVAE.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log \
--tensorboard _${PREFIX} --tensorboard-custom-fn $HOME/hww/incl-train/weaver-core/weaver/tensorboard_fn/JetClassII_evt_AD.py

// change to 10M QCD

PREFIX=evt_full.simTransformerVAE.unembed.beta0.heavyembed
config=$HOME/hww/incl-train/weaver-core/weaver/data_pheno/EvtClass/${PREFIX%%.*}.yaml
DATAFILE='/home/olympus/lyazj/software/ad-part/run/sm-qcd/data4/QCD/*.root'

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom \
--optimizer-option weight_decay 0.01 -o kld_weight 0 \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--gpus 3 --fetch-step 0.01 --num-workers 5 \
--samples-per-epoch $((512 * 1500)) --samples-per-epoch-val $((512 * 375)) \
--data-train $DATAFILE --data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/TransformerVAE.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log \
--tensorboard _${PREFIX} --tensorboard-custom-fn $HOME/hww/incl-train/weaver-core/weaver/tensorboard_fn/JetClassII_evt_AD.py

------------------------------------------
# 24.05.03 move to dijet anomaly
// now adding --seed, --extra-selection
NEVT=10000;

// PREFIX=Wkk_IAD.ensem20.512-188-64 network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(188,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.all --optimizer-option lr_mult ("model_ensemble.*ft_mlp\.[01].*",0.01) --start-lr 1e-3 --num-epochs 32 --lr-scheduler cosanneal --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_ak8puppi_AD_ensemble.py';
NEVT=10000; ## ~3.8sigma
NEVT=2400; ## ~1sigma
NEVT=0;
NEVT=500; ## ~0.2sigma, like the plots
NEVT=4800
# PREFIX=Wkk_IAD.ensem20.64-64.flat.nevt${NEVT} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';
# PREFIX=Wkk_IAD.ensem20.512-64-64.flat.nevt${NEVT} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --load-model-weights finetune_pheno.ensemble.0 --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';
PREFIX=Wkk_IAD.ensem20.512loadffn-lrmult0p01-64m.flat.nevt${NEVT} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(64,0.0)] -o merge_after_nth_layer 1 --load-model-weights finetune_pheno.ensemble.0 --optimizer-option lr_mult ("model_ensemble.*ft_mlp\.0.*",0.01) --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root";
GPU=0

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom --seed 42 \
--extra-selection "((event_class <= 18) | ((event_class == 21) & (event_no % 1000 < ($NEVT / 50))) )" \
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

## SALAD
NEVT=2400; ## ~1sigma
NEVT=500; ## ~0.2sigma, like the plots
NEVT=0;
PREFIX=Wkk_SALAD_step1.ensem20.64-64.flat.nevt${NEVT} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

wgtloss_opt=' -o num_ensemble_weight_model 20 -o ft_layer_params_weight_model [(64,0.0),(64,0.0)]'; wgtloss_path=$HOME/hww/incl-train/weaver-core/weaver/anomdet/model/Wkk_SALAD_step1.ensem20.64-64.flat.nevt${NEVT}/net_best_epoch_state.pt
PREFIX=Wkk_SALAD_step2.ensem20.64-64.flat.nevt${NEVT} network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 20 -o ft_layer_params [(64,0.0),(64,0.0)] -o merge_after_nth_layer 0  --load-model-weights wgtloss:'${wgtloss_path}' --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part0.root /mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_20ifb.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root /mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_20ifb.root";
GPU=0

## switch to loadffn
NEVT=500;
PREFIX=Wkk_SALAD_step1.ensem100.512-64-64.flat.nevt${NEVT} network_name=pheno/mlp_shared; ext_opt='-o num_ensemble 20 -o ft_layer_params [(512,0.0),(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --load-model-weights finetune_pheno.ensemble.0 --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

wgtloss_opt=' -o num_ensemble_weight_model 20 -o ft_layer_params_weight_model [(64,0.0),(64,0.0)]'; wgtloss_path=$HOME/hww/incl-train/weaver-core/weaver/anomdet/model/Wkk_SALAD_step1.ensem20.64-64.flat.nevt${NEVT}/net_best_epoch_state.pt
PREFIX=Wkk_SALAD_step2.ensem20.512-64-64.flat.nevt${NEVT} network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 100 -o use_mlp_shared True -o ft_layer_params [(64,0.0),(64,0.0)] -o merge_after_nth_layer 0 --load-model-weights wgtloss:'${wgtloss_path}' --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part0.root /mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_20ifb.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root /mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_20ifb.root";
GPU=3


python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom \
--extra-selection "((event_class <= 18) | ((event_class == 21) & (event_no % 1000 < ($NEVT / 50))) )" \
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


// implement run_salad.sh

./run_dijet_SALAD.sh Wkk 500 3 0

## IAD for low-level features
NEVT=20000; ## 8sigma
PREFIX=Wkk_IAD_lolv.nevt${NEVT} network_name=pheno/example_ParticleTransformer2023Dijet; ext_opt='--start-lr 1e-3 --num-epochs 50 --lr-scheduler none --samples-per-epoch 100000 --samples-per-epoch-val 100000 --batch-size 200 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_dijet_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.yaml";
rm -rf $HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_dijet/${PREFIX%%.*}.*auto*.yaml

DATAFILE_TRAIN="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part0.root";
DATAFILE_EVAL="/mldata/licq/datasets/JetClassII/sm_dijet/mixed_dijet_passsel_ntuple_merged_40ifb_part1.root";
GPU=2

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--extra-selection "((event_class <= 18) | ((event_class == 21) & (event_no % 1000 < ($NEVT / 50))) )" \
--use-amp --optimizer-option weight_decay 0.01 \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--optimizer ranger ${ext_opt} \
--gpus $GPU --fetch-step 0.01 \
--data-train $DATAFILE_TRAIN \
--data-val $DATAFILE_EVAL \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/anomdet/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto} --samples-per-epoch 200 --samples-per-epoch-val 200

# 24.05.04 finally: to full spectrum AD

run_weaver() {

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode custom \
--samples-per-epoch $((10 * 50000)) --samples-per-epoch-val $((10 * 50000)) \
--optimizer ranger ${ext_opt} \
--gpus $GPU \
--data-train $DATAFILE_TRAIN \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/anomdet/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/anomdet/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/anomdet/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto};

}
MASSRANGE=bin120to130
MASSRANGE=bin270to280
MASSRANGE=bin170to180
GPU=0
# PREFIX=fullspec_SALAD_step1_$MASSRANGE.ensem20.64-64.flat network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(64,0.0),(64,0.0)] --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model'; wgtloss_opt=' -o num_ensemble_weight_model 100 -o ft_layer_params_weight_model [(64,0.0),(64,0.0)]'; wgtloss_path=./model/$PREFIX/net_best_epoch_state.pt
PREFIX=fullspec_SALAD_step1_$MASSRANGE.ensem100.512-64.flat network_name=pheno/mlp; ext_opt='-o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.0 --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_fullspec_AD_ensemble.py --fetch-step 1 --in-memory'; wgtloss_opt=' -o num_ensemble_weight_model 100 -o ft_layer_params_weight_model [(512,0.0),(64,0.0)]'; wgtloss_path=./model/$PREFIX/net_best_epoch_state.pt

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_fullspec/${PREFIX%%.*}.yaml";
DATAFILE_TRAIN='/mldata/licq/datasets/JetClassII/mixed_ntuple/ntuples_*.root';

run_weaver

# PREFIX=${PREFIX/step1/step2} network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 100 -o ft_layer_params [(64,0.0),(64,0.0)] --load-model-weights wgtloss:'${wgtloss_path}' --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_fullspec_AD_ensemble.py';
PREFIX=${PREFIX/step1/step2}.ensem100.512-64.loadffn0-lrmult1 network_name=pheno/mlp_wgtloss; ext_opt=${wgtloss_opt}' -o num_ensemble 100 -o ft_layer_params [(512,0.0),(64,0.0)] --load-model-weights finetune_pheno.ensemble.0+wgtloss:'${wgtloss_path}' --optimizer-option lr_mult ("m_main.model_ensemble.*ft_mlp\.0.*",1) --start-lr 1e-3 --num-epochs 10 --lr-scheduler none --samples-per-epoch 500000 --samples-per-epoch-val 500000 --batch-size 10000 --train-mode-params metric:loss --use-last-model --tensorboard-custom-fn ../tensorboard_fn/JetClassII_fullspec_AD_ensemble.py';

config="$HOME/hww/incl-train/weaver-core/weaver/data_pheno/AD_fullspec/${PREFIX%%.*}.yaml";

run_weaver
