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

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' \
--use-amp --batch-size 512 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*100.root' \
--data-test \
'test:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/dnntuple_15898358-10.root' \
--samples-per-epoch $((15000 * 512)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root --in-memory

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger \
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
CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
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
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

##### (additional prediction)
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--gpus 2 \
--data-test \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--gpus 1 \
--data-test \
'ofcttbarfl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/*.root' \
'ofcttbarsl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## Then add LT+pixel hit pattern (smaller --fetch-step 0.005)
PREFIX=ak8_MD_inclv8_part_addltphp_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
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
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## Then add LT+pixel hit pattern - pixel only (no improvement w.r.t. +lost tracks)
PREFIX=ak8_MD_inclv8_part_addltphp_pixelonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## Then add LT+pixel hit pattern - with measurements only
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## Following above, but test using normalized pairs
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.normpair.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## with only wmeas+pixel+TIB
PREFIX=ak8_MD_inclv8_part_addltphp_wmeaspixeltibonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## Then add LT+pixel hit pattern (smaller --fetch-step 0.004)
PREFIX=ak8_MD_inclv8_part_addltphp_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3.try2
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

// okay fine... It seems like you cannot reproduce
// is it because of the fetch-step??

PREFIX=ak8_MD_inclv8_part_addltphp_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3.try4.fs0p08
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

// no it's not... we cannot reproduce the first result..
// ok it is because we are not switching back from normpair.....

### inference new Higgs 

NGPUS=1

for PREFIX in \
 ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addlosttrk_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addltphp_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addltphp_wmeaspixeltibonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3;
do \
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml; \
DATAPATH=/mldata/licq/deepjetak8; \
if [ "$PREFIX" == ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 ]; then three_coll=False; else three_coll=True; fi; \
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll $three_coll -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--gpus 2 \
--data-test \
'higlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*.root' \
'highi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*.root' \
'hwwlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-8_narrow/*.root' \
'hwwhi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-20_narrow/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root; \
done

for PREFIX in \
 ak8_MD_inclv8_part_addltphp_wmeasonly_manual.normpair.useamp.lite.gm5.ddp-bs768-lr6p75e-3 xx; \
do \
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml; \
DATAPATH=/mldata/licq/deepjetak8; \
if [ "$PREFIX" == ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 ]; then three_coll=False; else three_coll=True; fi; \
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll $three_coll -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--gpus 2 \
--data-test \
'higlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*.root' \
'highi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*.root' \
'hwwlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-8_narrow/*.root' \
'hwwhi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-20_narrow/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root; \
done

### inference new Higgs (second verison)

for PREFIX in \
 ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addlosttrk_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addltphp_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 \
 ak8_MD_inclv8_part_addltphp_wmeaspixeltibonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3;
do \
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml; \
DATAPATH=/mldata/licq/deepjetak8; \
if [ "$PREFIX" == ak8_MD_inclv8_part_2reg_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3 ]; then three_coll=False; else three_coll=True; fi; \
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll $three_coll -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 30 --optimizer ranger --fetch-step 0.008 \
--gpus 3 \
--data-test \
'hwwlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_LowPt_narrow/*.root' \
'hwwhi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_HighPt_narrow/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root; \
done


# 23.05.24 study the normpair!
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.normpair_v2.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(512,0.1)]' -o embed_dims '[64,256,64]' -o pair_embed_dims '[32,32,32]' -o pair_input_dim 6 \
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
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// the first normpair attempt fails... Why this cannot work...?

# 23.05.24 Now try large models

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.standardsize.gm5.ddp-bs512-lr4e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' \
--use-amp --batch-size 512 --start-lr 4e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
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

## even larger!!

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
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

## even larger with FC modified

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large_fc128-1024.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.005 \
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
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root --load-epoch 17

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large_fc64-1024.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

## FC modified to 64 - is it possible? (a small degradation)

NGPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(64,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.005 \
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
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root --load-epoch 10

## FC modified to 128 (perhaps our future choice), but use norm_pair

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.normpair.useamp.large_fc128-1024.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
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


# 22.06.17 export onnx
GPU=0
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid_output.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_opset17.onnx


PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.standardsize.gm5.ddp-bs512-lr4e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid_output.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model.onnx

/// 22.07.15 try out onnx with original file
GPU=0
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_opset_origfile.onnx


## test onnx scores
NGPUS=1
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}_inferonnxmodel.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 2 \
--data-test \
'hzzonnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 23.06.23 Try only using independent mH mW sample (exclude non 2DMesh sampple!)
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual_2dmeshsampleonly.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

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
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
'ofcttbarfl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/*.root' \
'ofcttbarsl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 23.06.27 Doing the AK15 training (norm pair, ParT-large with 128 FC)

## v8: get reweight factor

NGPUS=4
PREFIX=ak15_MD_inclv8_part_addltphp_wmeasonly
PREFIX=ak15_MD_inclv8_part_addltphp_wmeasonly.normpair.useamp.large_fc128-1024.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak15_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

## v8 test training
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp --batch-size 192 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak15_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*100.root' \
--data-test \
'test:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/dnntuple_15898358-10.root' \
--samples-per-epoch $((15000 * 512)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root --in-memory

## formal training
PREFIX=ak15_MD_inclv8_part_addltphp_wmeasonly.normpair.useamp.large_fc128-1024.gm5.ddp-bs192-lr1p5e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp --batch-size 192 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.004 \
--backend nccl --data-train \
$DATAPATH'/20230504_ak15_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'higgs2p:'$DATAPATHIFR'/20230504_ak15_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak15_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root --load-epoch 47

## == 11.11 switch back to AK15 temporarily to reproduce stage-2 tagger

PREFIX=ak15_MD_inclv8_part_addltphp_wmeasonly.useamp.large.gm5.ddp-bs192-lr1p5e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 192 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.004 \
--backend nccl --data-train \
$DATAPATH'/20230504_ak15_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak15_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'higgs2p:'$DATAPATHIFR'/20230504_ak15_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak15_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


## onnx export
/// for Run2 model, remember to replace einsum, and run with opset 11

GPU=0
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \ ## redo for stage-2
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_opset11.onnx

## test onnx sample
// batch size (=1?) has a slight influence to the score!
NGPUS=3
PREFIX=ak15_MD_inclv8_part_addltphp_wmeasonly.normpair.useamp.large_fc128-1024.gm5.ddp-bs192-lr1p5e-3
PREFIX=ak15_MD_inclv8_part_addltphp_wmeasonly.useamp.large.gm5.ddp-bs192-lr1p5e-3  ## redo for stage-2
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}_inferonnxmodel.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 \
--data-test \
'hzzonnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


# 23.08.18 back to AK8 

## testing new dataloader
PREFIX=ak15_MD_inclv8_part_addltphp_wmeasonly.TEST
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp --batch-size 192 --start-lr 1.5e-3 --num-epochs 50 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak15_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*100.root' \
--data-test \
'test:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/dnntuple_15898358-10.root' \
--samples-per-epoch $((1 * 512)) --samples-per-epoch-val $((1 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## export the normpair AK8 model

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.normpair.useamp.large_fc128-1024.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

/// for Run2 model, remember to replace einsum, and run with opset 11

GPU=0
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model.onnx

## test onnx sample
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}_inferonnxmodel.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 \
--data-test \
'hzzonnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(128,0),(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o norm_pair True \
--use-amp --batch-size 1 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 \
--data-test \
'hzzonnxtestbs1:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### test non-normpair version
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}_inferonnxmodel.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 200 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 \
--data-test \
'hzzonnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// found bug.. not masking particles when calculating sum of 4-vec!

### back to non-normpair AK8 model, export_embed = True
GPU=0
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o export_embed True \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_embed.onnx

# 23.08.29 stage-2 fine-tuning studies

## infer lastfc

PREFIX=ak8_lastfc_infer
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpu 3 \
--data-test \
'qcd-700to100:'$DATAPATHIFR'/20230826_ak8_UL17_v9/v9-AD/qcd-700to100.root' \
'mix_qcd-700to100_nsig100:'$DATAPATHIFR'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.root' \
'gghbb:'$DATAPATHIFR'/20230826_ak8_UL17_v9/v9-AD/GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8.root' \
'gghbb_nsig100:'$DATAPATHIFR'/20230826_ak8_UL17_v9/v9-AD/gghbb_nsig100.root' \
'gghbb_re2:'$DATAPATHIFR'/20230826_ak8_UL17_v9/v9-AD-re2/GluGluHToBB_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/finetune_stage2/example_ParticleTransformer2023Tagger_hybrid_lastfc.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/part_stage2_lastfc_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

PREFIX=ak8_lastfc_infer
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/${PREFIX%%.*}.incl.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpu 3 \
--data-test \
'gghww:'$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root' \
'gghwwsm:'$DATAPATH'/20230826_ak8_UL17_v9/GluGluHToWW_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8.root' \
'topww:'$DATAPATH'/20230826_ak8_UL17_v9/v9-topww/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root' \
'ofcttbarsl:'$DATAPATH'/20230826_ak8_UL17_v9/frompytorch/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/finetune_stage2/example_ParticleTransformer2023Tagger_hybrid_lastfc.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/part_stage2_lastfc_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpu 3 \
--data-test \
'mix_gghbb_msd100to140_seed42.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.0.root' \
'mix_gghbb_msd100to140_seed42.1:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.1.root' \
'mix_gghbb_msd100to140_seed42.2:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.2.root' \
'mix_gghbb_msd100to140_seed42.3:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.3.root' \
'mix_gghbb_msd100to140_seed42.4:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.4.root' \
'mix_gghbb_msd100to140_seed42.5:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.5.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/finetune_stage2/example_ParticleTransformer2023Tagger_hybrid_lastfc.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/part_stage2_lastfc_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## top fine-tuning

NGPUS=1
GPU=3
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_finetune_stage2_topVsQCD_manual.aux256-256; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] -o aux_layer_params [(256,0),(256,0)]'
PREFIX=ak8_finetune_stage2_topVsQCD_manual.aux256-256.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] -o concat_ft_layer 1 -o aux_layer_params [(256,0),(256,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/top/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --data-train \
$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--data-test \
'qcd:'$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATH'/20230826_ak8_UL17_v9/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 16 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// pure test
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU \
--data-test \
'qcd:'$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATH'/20230826_ak8_UL17_v9/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 16 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### also infer the original model, adding PNet non-MD!

NGPUS=1
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/top/${PREFIX%%.*}_infertopVsQCD.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 3 \
--data-test \
'qcd_finetune_stage2_topVsQCD:'$DATAPATH'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar_finetune_stage2_topVsQCD:'$DATAPATH'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 16 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root



## b-tau fine-tuning

### train a model including b-tau!

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_ftbtau.useamp.large.gm5.ddp-ngpu2-bs256-lr1p5e-3
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_ftbtau.useamp.large.gm5.ddp-ngpu3-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/btau/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

// pre-train
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger \
--gpus 1 --data-train \
$DATAPATH'/20230826_ak8_UL17_v9/PairVectorLQ_LQToBTau_HT-600to6000_M-15to250/*.root' \
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

// official training!!
NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--backend nccl --data-train \
$DATAPATH'/20230826_ak8_UL17_v9/PairVectorLQ_LQToBTau_HT-600to6000_M-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATH'/20230826_ak8_UL17_v9/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// for test only
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 0 \
--data-test \
'btau:'$DATAPATH'/20230826_ak8_UL17_v9/PairVectorLQ_LQToBTau_HT-600to6000_M-15to250/*.root' \n
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### fine-tune MLP for b-tau (no top categories, early version!)

NGPUS=1
GPU=2
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_finetune_stage2_btau_manual.aux256-256; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)]' // removed
PREFIX=ak8_finetune_stage2_btau_manual.aux256-256.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)' // removed
// testing lr
PREFIX=ak8_finetune_stage2_btau_manual.lr5e-3; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)]'
PREFIX=ak8_finetune_stage2_btau_manual.lr5e-3.dropout0p1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0.1)]' // removed
PREFIX=ak8_finetune_stage2_btau_manual.lr2e-3.dropout0p1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0.1)]'


config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/btau/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-3 --num-epochs 15 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus 0 --data-train \
$DATAPATH'/20230826_ak8_UL17_v9/PairVectorLQ_LQToBTau_HT-600to6000_M-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
--data-test \
'qcd:'$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'btau:'$DATAPATH'/20230826_ak8_UL17_v9/PairVectorLQ_LQToBTau_HT-600to6000_M-15to250/*.root' \
'higgs2p:'$DATAPATH'/20230826_ak8_UL17_v9/frompytorch/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 16 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// test only

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus 2 \
--data-test \
'higgs2p:'$DATAPATH'/20230826_ak8_UL17_v9/frompytorch/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'ttbar:'$DATAPATH'/20230826_ak8_UL17_v9/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 16 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### fine-tune MLP for b-tau (+top cat)
// adding ttbarvm for training + ttbar for test

NGPUS=1
GPU=2
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_finetune_stage2_btauwtop_manual.lr5e-3.ft1024-1024; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0.1),(1024,0)]'
PREFIX=ak8_finetune_stage2_btauwtop_manual.lr5e-3.try2; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)]'
PREFIX=ak8_finetune_stage2_btauwtop_manual.lr5e-3.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'
PREFIX=ak8_finetune_stage2_btauwtop_manual.lr5e-3.aux256.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0),(256,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/btau/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-3 --num-epochs 15 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus 1 --data-train \
$DATAPATH'/20230826_ak8_UL17_v9/PairVectorLQ_LQToBTau_HT-600to6000_M-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/frompytorch/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
--data-test \
'qcd:'$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'btau:'$DATAPATH'/20230826_ak8_UL17_v9/PairVectorLQ_LQToBTau_HT-600to6000_M-15to250/*.root' \
'higgs2p:'$DATAPATH'/20230826_ak8_UL17_v9/frompytorch/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'ttbar:'$DATAPATH'/20230826_ak8_UL17_v9/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 16 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


### [test] fine-tune MLP without b-tau

NGPUS=1
GPU=2
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_finetune_stage2_nobtau_manual.lr5e-3.dropout0p1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0.1)]'
PREFIX=ak8_finetune_stage2_nobtau_manual.fullcopy; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0.1)] --load-model-weights finetune_stage2.exactcopy.higgs+qcd'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/btau/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 0 --num-epochs 15 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus 0 --data-train \
$DATAPATH'/20230826_ak8_UL17_v9/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
--data-test \
'qcd:'$DATAPATH'/20230826_ak8_UL17_v9/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'higgs2p:'$DATAPATH'/20230826_ak8_UL17_v9/frompytorch/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 16 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


## gghww mass regression

### first approach
// note that it is necessary to append "--fetch-by-files --fetch-step 1" in case you use "--in-memory" !!
NGPUS=1
GPU=0
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_finetune_stage2_gghww_manual.lr2e-2.aux32-32;  network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] -o aux_layer_params [(32,0),(32,0)] -o loss_type "reg" --load-model-weights finetune_stage2.exactcopy.res_mass --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'; data_train=$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root'
PREFIX=ak8_finetune_stage2_gghww_manual.addbkg.lr2e-2.aux32-32;  network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] -o aux_layer_params [(32,0),(32,0)] -o loss_type "reg" --load-model-weights finetune_stage2.exactcopy.res_mass --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'; data_train=$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root '$DATAPATH'/20230826_ak8_UL17_v9/v9-topww/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/gghww/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode regression \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --data-train $data_train --fetch-by-files --fetch-step 1 --in-memory \
--data-test \
'gghww:'$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root' \
'ofcttbarsemilep:'$DATAPATH'/20230826_ak8_UL17_v9/v9-topww/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// predict

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode regression --predict \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU \
--data-test \
'gghww:'$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root' \
'gghwwsm:'$DATAPATH'/20230826_ak8_UL17_v9/GluGluHToWW_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8.root' \
'ofcttbarsemilep:'$DATAPATH'/20230826_ak8_UL17_v9/v9-topww/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root' \
'ofcttbarsl:'$DATAPATH'/20230826_ak8_UL17_v9/frompytorch/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### 23.11.16 redo for AN
NGPUS=1
GPU=3
DATAPATH=/mldata/licq/deepjetak8

PREFIX=ak8_finetune_stage2_gghww_ratio_manual.lr5e-4.aux32-32;  network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] -o aux_layer_params [(32,0),(32,0)] -o loss_type "reg" --load-model-weights finetune_stage2.exactcopy.res_mass --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'; data_train=$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root'
PREFIX=ak8_finetune_stage2_gghww_ratio_manual.lr5e-4.aux32-32.unfreezemlp;  network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] -o aux_layer_params [(32,0),(32,0)] -o loss_type "reg" --load-model-weights finetune_stage2.exactcopy.res_mass'; data_train=$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root'
PREFIX=ak8_finetune_stage2_gghww_ratio_manual.lr5e-4.aux256-256.unfreezemlp;  network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] -o aux_layer_params [(256,0),(256,0)] -o loss_type "reg" --load-model-weights finetune_stage2.exactcopy.res_mass'; data_train=$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root'

PREFIX=ak8_finetune_stage2_gghww_ratio_noAux_manual.lr5e-4.unfreezemlp;  network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] -o loss_type "reg" --load-model-weights finetune_stage2.exactcopy.res_mass'; data_train=$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/gghww/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode regression \
--batch-size 512 --start-lr 5e-4 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --data-train $data_train --fetch-by-files --fetch-step 1 --in-memory \
--data-test \
'gghww:'$DATAPATH'/20230826_ak8_UL17_v9/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass.root' \
'gghwwsm:'$DATAPATH'/20230826_ak8_UL17_v9/GluGluHToWW_Pt-200ToInf_M-125_TuneCP5_MINLO_13TeV-powheg-pythia8.root' \
'topww:'$DATAPATH'/20230826_ak8_UL17_v9/v9-topww/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


## AD finetuning

GPU=3
DATAPATH=/data/pubfs/licq/deepjetak8
PREFIX=ak8_anomdat.lr2e-2.massnorm; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)]'
PREFIX=ak8_anomdat.lr2e-2.massnorm.256-256; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(256,0),(256,0)]'
PREFIX=ak8_anomdat.lr2e-2.massnorm.drop0p2; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0.2)]'
PREFIX=ak8_anomdat.lr2e-2.massnorm.1y; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params []'
PREFIX=ak8_anomdat.lr2e-2.massnorm.loadparams; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] --load-model-weights finetune_stage2_AD'
PREFIX=ak8_anomdat.lr2e-2.massnorm.wd0p1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] --optimizer-option weight_decay 0.1'
PREFIX=ak8_anomdat.lr2e-2.massnorm.loadparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] --load-model-weights finetune_stage2_AD --optimizer-option lr_mult ("ft_mlp.0.*",0.00001)'
PREFIX=ak8_anomdat.lr2e-2.massnorm.loadparamsfreeze.1024-1024; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(1024,0)] --load-model-weights finetune_stage2_AD --optimizer-option lr_mult ("ft_mlp.0.*",0.00001)'

PREFIX=ak8_anomdat.lr2e-2.massnorm; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)]'
PREFIX=ak8_anomdat.lr2e-2.massnorm.wd1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] --optimizer-option weight_decay 1'
PREFIX=ak8_anomdat.lr5e-3.massnorm.loadparamsfreeze.wd2; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] --optimizer-option weight_decay 2 --load-model-weights finetune_stage2.0 --optimizer-option lr_mult ("ft_mlp.0.*",1e-10)'
PREFIX=ak8_anomdat.lr2e-2.massnorm.loadallparamsfreeze.wd0p1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] --optimizer-option weight_decay 0.1 --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'
PREFIX=ak8_anomdat.lr2e-2.massnorm.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-2 --num-epochs 50 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.2.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.3.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.4.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.5.root' \
--data-val \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.1.root' \
--data-test \
'mix_qcd-700to100_nsig100.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.0.root' \
--samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// conclusion
 - dropout helps slightly
 - load paramters / load + freeze coverge slowly (???), but same effect => pre-load does not help, but why

### train on merge data, with kfold=6
// generate new yamls with different kfold to run the code.

GPU=3
DATAPATH=/data/pubfs/licq/deepjetak8
PREFIX=ak8_anomdat.mix_qcd-700to100_nsig100.massnorm; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)]'
PREFIX=ak8_anomdat.mix_qcd-700to100_nsig100.massnorm.loadparamsfreeze.wd2; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)] --optimizer-option weight_decay 2 --load-model-weights finetune_stage2.0 --optimizer-option lr_mult ("ft_mlp.0.*",1e-10)'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/AD/${PREFIX%%.*}.yaml

filepath=$DATAPATH'/20230826_ak8_UL17_v9/v9-AD/mix_qcd-700to100_nsig100.root'
for kfold in `seq 0 5`; do
kfoldval=$(( (kfold + 1) % 6 ))
data_train=""; for k in `seq 0 5`; do if (( (k != kfold) && (k != kfoldval) )); then data_train+="${filepath%.root}.$k.root "; fi; done
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 50 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $data_train \
--data-val ${filepath%.root}.$kfoldval.root \
--data-test \
'mix_qcd-700to100_nsig100:'${filepath%.root}.$kfold.root \
--samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}.$kfold/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX.$kfold/pred.root ;
done

// changed to 5e-3

## AD finetuning attempt 2

GPU=3
DATAPATH=/mldata/licq/deepjetak8

PREFIX=ak8_finetune_stage2_anomdat.lr2e-2.massnorm; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0)]'
PREFIX=ak8_finetune_stage2_anomdat.lr2e-2.massnorm.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'
PREFIX=ak8_finetune_stage2_anomdat.ft256-256.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0),(256,0),(256,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",1e-10)'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/AD/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re/mix_all_msd100to140_seed42.2.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re/mix_all_msd100to140_seed42.3.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re/mix_all_msd100to140_seed42.4.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re/mix_all_msd100to140_seed42.5.root' \
--data-val \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re/mix_all_msd100to140_seed42.1.root' \
--data-test \
'mix_all_msd100to140_seed42.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re/mix_all_msd100to140_seed42.0.root' \
--samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

## AD finetuning attempt 3 231203


GPU=3
DATAPATH=/mldata/licq/deepjetak8

PREFIX=ak8_finetune_stage2_anomdet_std.ft256; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(256,0)]'
PREFIX=ak8_finetune_stage2_anomdet_cen.ft256; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(256,0)]'
PREFIX=ak8_finetune_stage2_anomdet_std.ft256.lrschnone; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(256,0)] --lr-scheduler none'
PREFIX=ak8_finetune_stage2_anomdet_std.ft256.dp0p5; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(256,0.5)]'
PREFIX=ak8_finetune_stage2_anomdet_std.noft; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params []'

PREFIX=ak8_finetune_stage2_anomdet_cen.loadallparamsfreeze; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0)'
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.wd1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --optimizer-option weight_decay 1'

PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.wd1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314")] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --optimizer-option weight_decay 1' ## wierd distribution, but seems sucessful!
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.wd10; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314")] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --optimizer-option weight_decay 10'

PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.mseloss; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314")] -o loss_type "cls:mse" --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0)'
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.mseloss.lrschnone; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314")] -o loss_type "cls:mse" --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --lr-scheduler none --start-lr 1e-2' # half successful
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.mseloss.lrschnone.ep1-s1300000; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314")] -o loss_type "cls:mse" --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --lr-scheduler none --start-lr 1e-2 --num-epochs 1 --samples-per-epoch 1300000'
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.mseloss.lrschnone.lr1e-3.ep1-s1300000; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314")] -o loss_type "cls:mse" --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --lr-scheduler none --start-lr 1e-3 --num-epochs 1 --samples-per-epoch 1300000'

PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.lrschnone; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314")] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --lr-scheduler none --start-lr 1e-2'
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.mseloss.lrschnone; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0)] -o loss_type "cls:mse" --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --lr-scheduler none --start-lr 1e-2'

PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.idxHbbvsQCD.mseloss.lrschnone; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314","idx:17,309:313")] -o loss_type "cls:mse" --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0) --lr-scheduler none --start-lr 1e-2'


#### we should not trust in those not-fully-trained versions... back to the original ones
PREFIX=ak8_finetune_stage2_anomdet_bi.ft256; network_name=finetune_stage2/mlp; ext_opt='--train-mode-params metric:loss -o ft_layer_params [(256,0)]'
PREFIX=ak8_finetune_stage2_anomdet_bi.ft256.lr2e-2; network_name=finetune_stage2/mlp; ext_opt='--train-mode-params metric:loss -o ft_layer_params [(256,0)] --start-lr 2e-2'
PREFIX=ak8_finetune_stage2_anomdet_bi.ft256.lr5e-3; network_name=finetune_stage2/mlp; ext_opt='--train-mode-params metric:loss -o ft_layer_params [(256,0)] --start-lr 5e-3'
PREFIX=ak8_finetune_stage2_anomdet_cen.ft256.lrschcosanneal; network_name=finetune_stage2/mlp; ext_opt='--train-mode-params metric:loss -o ft_layer_params [(256,0)] --lr-scheduler cosanneal'

PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.idxHbbvsQCD; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314","idx:17,309:314")] --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0)'
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.idxHbbvsQCDothers; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314","idx:17,313")] --load-model-weights finetune_stage2.all --optimizer-option freeze "ft_mlp.[01].*"'
PREFIX=ak8_finetune_stage2_anomdet_bi.loadallparamsfreeze.sm.idxHbbvsQCD.mseloss; network_name=finetune_stage2/mlp; ext_opt='--train-mode-params metric:loss -o ft_layer_params [(1024,0),(316,0,"sm:0:314","idx:17,309:314")] -o loss_type "cls:mse" --load-model-weights finetune_stage2.all --optimizer-option lr_mult ("ft_mlp.[01].*",0)' # not working at all


config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/ADre2/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle --samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.2.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.3.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.4.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.5.root' \
--data-val \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.1.root' \
--data-test \
'mix_all_msd100to140_seed42.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.0.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// predict only
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle --samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-test \
'mix_epoch29_gghbb_msd100to140_seed42.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_msd100to140_seed42.0.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_epoch-29_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

#### divide qcd with 25 to remove depulicated events.. This seems to be an important issue

PREFIX=ak8_finetune_stage2_anomdet_bi.qcddiv25.loadallparamsfreeze.sm.idxHbbvsQCDothers; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(1024,0),(316,0,"sm:0:314","idx:17,313")] --load-model-weights finetune_stage2.all --optimizer-option freeze "ft_mlp.[01].*"'
PREFIX=ak8_finetune_stage2_anomdet_bi.qcddiv25.ft256; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(256,0)]'
PREFIX=ak8_finetune_stage2_anomdet_bi.qcddiv25.ft64-64-64; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)]'
PREFIX=ak8_finetune_stage2_anomdet_bi.qcddiv25.ft64-64-64.dp0p7; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 2e-2' // works so good!!

PREFIX=ak8_finetune_stage2_anomdet_bi.qcddiv25.ft64-64-64.dp0p7.try2; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop' // this is a mature state
PREFIX=ak8_finetune_stage2_anomdet_std.qcddiv25.ft64-64-64.dp0p7.try2; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop' 

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/ADre2/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle --samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42.2.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42.3.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42.4.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42.5.root' \
--data-val \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42.1.root' \
--data-test \
'mix_gghbb_qcddiv25_msd100to140_seed42.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42.0.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

// predict only
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle --samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-test \
'mix_gghbb_qcddiv25_msd100to140_seed42.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42.0.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_epoch-32_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

#### official gghbb launching

PREFIX=ak8_finetune_stage2_anomdet_bi.qcddiv25.ft64-64-64.dp0p7.ofc1; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop' 

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/ADre2/${PREFIX%%.*}.yaml

filepath=$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghbb_qcddiv25_msd100to140_seed42'
for kfold in `seq 0 5`; do
kfoldval=$(( (kfold + 1) % 6 ))
data_train=""; for k in `seq 0 5`; do if (( (k != kfold) && (k != kfoldval) )); then data_train+="$filepath.$k.root "; fi; done
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle --samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train $data_train \
--data-val $filepath.$kfoldval.root \
--data-test \
'mix_gghbb_qcddiv25_msd100to140_seed42.'$kfold':'$filepath.$kfold.root \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}.$kfold/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX.$kfold/pred.root ;
done


#### now trying gghww
PREFIX=ak8_finetune_stage2_anomdet_bi.qcddiv25.ft64-64-64.dp0p7.gghww; network_name=finetune_stage2/mlp; ext_opt='-o ft_layer_params [(64,0.7),(64,0.7),(64,0.7)] --start-lr 5e-3 --num-epochs 64 --lr-scheduler cosanneal --train-mode-params metric:loss --early-stop'

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_stage2/ADre2/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle --samples-per-epoch $((100 * 512)) --samples-per-epoch-val $((100 * 512)) \
${ext_opt} \
--gpus $GPU --fetch-by-files --fetch-step 1 --in-memory \
--data-train \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghww_qcddiv25_msd100to140_seed42.2.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghww_qcddiv25_msd100to140_seed42.3.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghww_qcddiv25_msd100to140_seed42.4.root' \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghww_qcddiv25_msd100to140_seed42.5.root' \
--data-val \
$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghww_qcddiv25_msd100to140_seed42.1.root' \
--data-test \
'mix_gghww_qcddiv25_msd100to140_seed42.0:'$DATAPATH'/20230826_ak8_UL17_v9/v9-AD-re2/mix_gghww_qcddiv25_msd100to140_seed42.0.root' \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


// this is also problematics... Should not use "--fetch-by-files --fetch-step 1 --in-memory" because it only loads 1 file!!