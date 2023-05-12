# >>> Continue with stage-2 development

# 23.03.08 Finner bins with PNet

NGPUS=1
PREFIX=ak8_MD_vminclv2_pre2_stage2_finemass
PREFIX=ak8_MD_vminclv2_pre2_stage2_massbinw20
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_stage2_finemass
PREFIX=ak8_MD_vminclv2_pre2_stage2_massbinw20

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

## test set
DATAPATH=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww:'$DATAPATH'/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


# 23.03.16 Train with HWW 2DMesh

NGPUS=1
PREFIX=ak8_MD_vminclv2_pre2_stage2_hww2dmesh
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_stage2_hww2dmesh
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}


## test set for XWW:
GPU=1
PREFIX=ak8_MD_vminclv2_pre2
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'xww:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


# 23.03.17 Train with HWW + HWW 2DMesh

NGPUS=1
PREFIX=ak8_MD_vminclv2_pre2_stage2_hww2sets
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_stage2_hww2sets
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

## test set
DATAPATH=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww:'$DATAPATH'/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xqq:'$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## test set for other two HWW samples: for all three model
DATAPATH=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'xwwfixr:'$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


# 23.03.24 HWW three training again... Develop in v6 condition...!
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_origrecast.try2
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## get reweighting factors
NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xwwfixr:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
'hqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATHIFR'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATHIFR'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## HWW 2DMesh

## get reweighting factors

NGPUS=2
PREFIX=ak8_MD_vminclv2_pre2_origrecast_stage2_hww2dmesh
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8


NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
NGPUS=2
PREFIX=ak8_MD_vminclv2_pre2_origrecast_stage2_hww2dmesh.farm221 ## try on farm221
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/data/licq/samples/deepjetak8

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 1024 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--data-test \
'hww:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xwwfixr:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## formal training (back to zeus)
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_origrecast_stage2_hww2dmesh
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--load-epoch 14 \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xwwfixr:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
'hqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATHIFR'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATHIFR'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


## just test
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xwwfixr:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## HWW+XWW

## get reweighting factors

PREFIX=ak8_MD_vminclv2_pre2_origrecast_stage2_hww2dsets
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xwwfixr:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
'hqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATHIFR'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATHIFR'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## ==finish== test original pnet model
PREFIX=ak8_MD_vminclv2_pre2_origrecast
MODELPREFIX=ak8_MD_vminclv2_pre2
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xwwfixr:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
'hqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATHIFR'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATHIFR'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$MODELPREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$MODELPREFIX/v6/pred.root

## inference of HWW2DMesh for gen study
PREFIX=ak8_MD_vminclv2_pre2_origrecast
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}_inferhww2dmesh.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'xww2dmesh_genvar:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## inference of HWW2DMesh for gen study
PREFIX=ak8_MD_vminclv2_pre2_origrecast
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}_inferhww2dmesh.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'xww2dmesh_genvar:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 23.03.30 HWW with WW balance

## get reweighting factors

PREFIX=ak8_MD_vminclv2_pre2_origrecast_stage2_hww2dsetswwbal
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## training

NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xww:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_JHUGen_MX-600to6000_MH-15to250_v2/*/*0.root' \
'xwwfixr:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*0.root' \
'xww2dmesh:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*/*0.root' \
'hqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATHIFR'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATHIFR'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATHIFR'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 6 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 23.04.02 try origrecast inlcuding top w/ unmerged b

NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_origrecast_stage2_unmergedtopb
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## get reweighting factors
NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## formal training
NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--data-test \
'hww:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## inference: SM ttbar
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_origrecast_stage2_unmergedtopb
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}_infer.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'ofcttbarsemilep:'$DATAPATH'/20221023_ak8_UL17_v6/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
'ofcttbarfulllep:'$DATAPATH'/20221023_ak8_UL17_v6/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## inference of origrecast on ttbar samples, including unmerged-b
PREFIX=ak8_MD_vminclv2_pre2_origrecast
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}_inferttbar.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'ttbarincl:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 23.05.04: fully inclusive! dnntuplesv7

NGPUS=4
PREFIX=ak8_MD_inclv7_pn_recast
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## get reweighting factors
NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230423_ak8_UL17_v7/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230423_ak8_UL17_v7/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230423_ak8_UL17_v7/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230423_ak8_UL17_v7/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230423_ak8_UL17_v7/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230423_ak8_UL17_v7/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230423_ak8_UL17_v7/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230423_ak8_UL17_v7/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## for test!

PREFIX=ak8_MD_inclv8_pn_originput
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 \
--batch-size 512 --start-lr 5e-3 --num-epochs 1 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*00.root' \
--data-test \
'test:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/dnntuple_15898358-10.root' \
--samples-per-epoch $((5 * 512)) --samples-per-epoch-val $((5 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## v8: get reweight factor

NGPUS=4
PREFIX=ak8_MD_inclv8_pn_originput
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_dim 512 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## test the utility
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 \
--batch-size 512 --start-lr 5e-3 --num-epochs 1 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*00.root' \
--data-test \
'test:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/dnntuple_15898358-10.root' \
--samples-per-epoch $((500 * 512)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## v8: formal training
### batchsize=768, lr=8e-3

PREFIX=ak8_MD_inclv8_pn_originput.bs768
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
--log-dir $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}_torchrun \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 -o fc_dim 512 --compile-model \
--batch-size 768 --start-lr 8e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 2023.05.08: use ParT to study input change!

PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## v8 test training
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' \
--use-amp --batch-size 768 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*100.root' \
--data-test \
'test:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/dnntuple_15898358-10.root' \
--samples-per-epoch $((15000 * 512)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root --in-memory

## v8 formal training (interupted.. wrong fj_mass config)
NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' \
--use-amp --batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

##### modify to non-dist training: change --gpus, modify bs and lr, change --num-workers (not used!!)
NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' \
--use-amp --batch-size 1280 --start-lr 2.5e-3 --num-epochs 50 --optimizer ranger \
--gpus 0,2,3 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 20 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## v8: try a liter ParT trainning.. (now double bs to 512!)
PREFIX=ak8_MD_inclv8_part_2reg_manual.lite
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--batch-size 1024 --start-lr 2e-3 --num-epochs 50 --optimizer ranger \
--gpus 0,2,3 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 20 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## v8: try a liter ParT trainning.. with amp!
// (now correct the LR: for DP, scale(double) the LR; for DDP, scale the size..)

PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.ddp-bs1024-lr3e-3
PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.ddp-bs768-lr2p25e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 2.25e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## This is the formal one!!

///PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.bs2048-lr2e-3
///PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.bs1536-lr1e-3
///PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.dp-bs3072-lr1e-3
PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.dp-bs2048-lr0p6e-3.try4
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8


NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 2048 --start-lr 0.6e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--gpus 0,2,3 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 30 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// now run predict routine to get the embedding space
NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 1 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 512 --start-lr 0.6e-3 --num-epochs 31 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*100.root' \
--data-val \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--samples-per-epoch $((1 * 512 / $NGPUS)) --samples-per-epoch-val $((100 * 512)) \
--data-config ${config} --num-workers 12 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}.test_embed/net --load-epoch 29 \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/embed.log --tensorboard _${PREFIX}.test_embed

## Tune paramters (note: change to ParT2023_orig)

PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.dp-bs2048-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 2048 --start-lr 2e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--gpus 1,2,3 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 30 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## Tune paramters (note: change to ParT2023_orig, use DDP)

PREFIX=ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13##TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 23.05.12 Study input variables with ParT-lite

PREFIX=ak8_MD_inclv8_part_addlosttrk_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## v8 test training
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*100.root' \
--data-test \
'test:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/dnntuple_15898358-10.root' \
--samples-per-epoch $((15000 * 512)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root --in-memory

## Formal training

NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13##TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root
