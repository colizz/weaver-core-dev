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

# try 50 epochs?

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

## 24.02.01 re-train with corrected puppi tune and AK8 jets

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