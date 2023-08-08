
# recap previous Dawei's training with new QCD sample
NGPUS=1
PREFIX=ak8_MD_split_cq ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
'/home/olympus/licq/data/hww/tmp/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravToWWToWhadWhad_narrow_M-500to8000_TuneCP5/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log /home/olympus/licq/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log --tensorboard _${PREFIX}

## pure test
NGPUS=2
PREFIX=ak8_MD_addHqq_pre1 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --gpus 1 \
--batch-size 512 --start-lr 1e-2 --num-epochs 30 --optimizer ranger \
--data-train \
'/home/olympus/licq/data/hww/tmp/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*00.root' \
'/home/olympus/licq/data/hww/tmp/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*00.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*00.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*0.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravToWWToWhadWhad_narrow_M-500to8000_TuneCP5/*/*00.root' \
--samples-per-epoch $((7500 * 512)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net

# now start
# but first, let's generate reweighting map with full events

NGPUS=1
PREFIX=ak8_MD_addHqq_pre1 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
'/home/olympus/licq/data/hww/tmp/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'/home/olympus/licq/data/hww/tmp/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravToWWToWhadWhad_narrow_M-500to8000_TuneCP5/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log /home/olympus/licq/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log --tensorboard _${PREFIX}

NGPUS=1
PREFIX=ak8_MD_addHqq_pre2 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'/mldata/licq/deepjetak8/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravToWWToWhadWhad_narrow_M-500to8000_TuneCP5/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log /home/olympus/licq/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log --tensorboard _${PREFIX}

# formal training
NGPUS=2
PREFIX=ak8_MD_addHqq_pre2 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'/mldata/licq/deepjetak8/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravToWWToWhadWhad_narrow_M-500to8000_TuneCP5/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log /home/olympus/licq/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

############
# now starts inference
# HWW: higgs signal vs. QCD
GPU=0
PREFIX=ak8_MD_split_cq ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_infer.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'hww:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/fudawei/tagger/boosted-HWW/model/$PREFIX/_best_epoch_state.pt \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 3 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# HWW previous v1 (fake)
GPU=0
PREFIX=ak8_MD_sdmass_5PIDvar_previous ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_infer.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'hww:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/fudawei/tagger/boosted-HWW/model/$PREFIX/_best_epoch_state.pt \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 6 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# real v1
GPU=0
PREFIX=ak8_MD_HWWv1 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/ak8_MD_sdmass_5PIDvar_previous_infer.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'hww:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/fudawei/weaver/output/hwwVsQCD/pnet.hwwVsQCD.sdmass.fine.addis.20million/pnet.hwwVsQCD.sdmass.fine.addis.20million \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 6 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# real v1 updated (0921, with new v1 training from Dawei)
GPU=0
PREFIX=ak8_MD_HWWv1_retrain ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/ak8_MD_sdmass_5PIDvar_previous_infer.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'hww:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/fudawei/tagger/boosted-HWW/model/ak8_MD_v1/_best_epoch_state.pt \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 6 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


# Hbb: original higgs->bb sig vs. QCD
GPU=3
PREFIX=ak8_points_pf_sv_mass_decorr ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/legacy/${PREFIX//./_}.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'xqq:/mldata/licq/deepjetak8/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'hqq:/mldata/licq/deepjetak8/20200601_ak8_UL17/GluGluToBulkGravitonToHHTo4B_M-1000_narrow_WZHtag_TuneCP5_PSWeights_13TeV-madgraph-pythia8/*/*/*/*.root' \
'qcd:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/legacy/ParticleNetMD_UL.pt \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 6 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# latest incl sample
GPU=1
PREFIX=ak8_MD_addHqq_pre2 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_infer2.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'xqq:/mldata/licq/deepjetak8/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'hww:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 6 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

### 22.10.01
# Regresion

### test
NGPUS=2
PREFIX=ak8_MD_hwwOnly ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/regression/${PREFIX//./_}.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --regression-mode \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_mass_regression.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log /home/olympus/licq/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

# formal run
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --regression-mode \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_mass_regression.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log /home/olympus/licq/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log  --tensorboard _${PREFIX}

# prediction
GPU=0
PREFIX=ak8_MD_hwwOnly ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/regression/${PREFIX//./_}_infer.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --train-mode regression --predict \
--data-test \
'hww:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 3 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


## 22.10.06

# get factors first
NGPUS=1
PREFIX=ak8_MD_vmincl_pre3 ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log


# formal training
NGPUS=4
PREFIX=ak8_MD_vmincl_pre3 ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

# predict
GPU=0
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_infer.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'xqq:/mldata/licq/deepjetak8/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'hww:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 6 \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


### 10.08 test training on ParT

NGPUS=1
PREFIX=ak8_MD_vminclParT_test ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp --batch-size 512 --start-lr 1e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net


## 10.10 test cls+regression

NGPUS=1
PREFIX=ak8_MD_vmincl_pre3_addreg_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
'/mldata/licq/deepjetak8/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*00.root' \
--samples-per-epoch $((5 * 512 / $NGPUS)) --samples-per-epoch-val $((5 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix /home/olympus/licq/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net

## formal training on ParticleNet
NGPUS=4
PREFIX=ak8_MD_vmincl_pre3_addreg_manual ## needs modification
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

# predict
GPU=0
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

 # hqq is added after v6
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

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww:'$DATAPATH'/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww:'$DATAPATH'/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


## 10.12 finally, comes the last story
# get reweight factors
NGPUS=1
PREFIX=ak8_MD_vminclv2_pre2 ## needs modification
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

# formal training
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2 ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_inferhqq.yaml
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

## actually also rerun it in the original mode (10.25)
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_noreg_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

# and for inference
DATAPATH=/data/pubfs/licq/deepjetak8

## 10.13 export ParT model...
## first let's test previous ParT

GPU=0
PREFIX=ak8_MD_vminclParT_addreg_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid_outputWithHidNeurons.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_opset14.onnx

# also try exporting previous v1 model again, with the new config
GPU=0
PREFIX=ak8_MD_HWWv1 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/ak8_MD_sdmass_5PIDvar_previous_infer.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/fudawei/weaver/output/hwwVsQCD/pnet.hwwVsQCD.sdmass.fine.addis.20million/pnet.hwwVsQCD.sdmass.fine.addis.20million_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model.onnx


## okay finally we do some onnx val
GPU=0
PREFIX=ak8_MD_vminclParT_addreg_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww_onnxtest:/home/olympus/licq/hww/incl-train/data/output_onnxtest.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

GPU=0
PREFIX=ak8_MD_HWWv1 ## needs modification
config=/home/olympus/licq/hww/incl-train/weaver-core/weaver/data_new/incl/ak8_MD_sdmass_5PIDvar_previous_infer.yaml

python /home/olympus/licq/hww/incl-train/weaver-core/weaver/train.py --predict \
--data-test \
'hww_onnxtest:/home/olympus/licq/hww/incl-train/data/output_onnxtest.root' \
--data-config ${config} \
--network-config /home/olympus/licq/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix /home/olympus/fudawei/weaver/output/hwwVsQCD/pnet.hwwVsQCD.sdmass.fine.addis.20million/pnet.hwwVsQCD.sdmass.fine.addis.20million \
--gpus ${GPU} --batch-size 512 --start-lr 5e-3 --num-workers 6 \
--predict-output /home/olympus/licq/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# so you know it's because wrap?
GPU=0
PREFIX=ak8_MD_vminclParT_addreg_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_fixwrap.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'hww_onnxtest_fixwrap:/home/olympus/licq/hww/incl-train/data/output_onnxtest.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## one last step! exporting onnx...
GPU=0
PREFIX=ak8_MD_vminclv2ParT_manual_fixwrap
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_inferhqq.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid_outputWithHidNeurons.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_best_epoch_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model.onnx


##################################
# fine-tune
NGPUS=1
PREFIX=ak8_rawscores_topVsQCD_manual_pre1.0layer ## needs modification
PREFIX=ak8_hidneurons_topVsQCD_manual_pre1.0layer ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 30 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/mlp.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

GPU=0
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/mlp.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# try 2-path mlp
NGPUS=1
PREFIX=ak8_rawscores_topVsQCD_manual_pre2 ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger \
--gpus 2 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 30 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/mlp_2p.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

# combined version
NGPUS=1
GPU=3
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_hidneurons_topVsQCD_manual_pre1.0layer; network_name=mlp; ext_opt= # raw score, two path (stoped) ## needs modification
PREFIX=ak8_hidneurons_topVsQCD_manual_pre1_rmbasic.0layer.onecycle; network_name=mlp; ext_opt= # raw score, two path (stoped) ## needs modification
PREFIX=ak8_rawscores_topVsQCD_manual_pre2; network_name=mlp_2p; ext_opt= # raw score, two path (stoped)
PREFIX=ak8_rawscores_topVsQCD_manual_pre1.candle; network_name=fintune_test/candle; ext_opt= # try out candle, without any learnable params.. failed
PREFIX=ak8_hidneurons_topVsQCD_manual_pre2.gated; network_name=fintune_test/mlp_2p_gated; ext_opt= # first try using gates. killed (lr not properly set?)
PREFIX=ak8_hidneurons_topVsQCD_manual_pre2.gated; network_name=fintune_test/mlp_2p_gated; ext_opt= # second try # sucess! a benchmark indeed
PREFIX=ak8_hidneurons_topVsQCD_manual_pre2.gated.neurons_in_preprocess; network_name=fintune_test/mlp_2p_gated; ext_opt="-o neurons_in_preprocess True" # third try (even better performance, good)
PREFIX=ak8_hidneurons_topVsQCD_manual_pre2.gated.neurons_in_preprocess.mult; network_name=fintune_test/mlp_2p_gated; ext_opt="-o neurons_in_preprocess True" # not working at all!
PREFIX=ak8_hidneurons_topVsQCD_manual_pre2_extfeat.gated.neurons_in_preprocess; network_name=fintune_test/mlp_2p_gated; ext_opt="-o neurons_in_preprocess True" # adding extra jet feats: similar performance

PREFIX=ak8_hidneurons_topVsQCD_manual_pre2.gated_on_hid.neurons_in_preprocess; network_name=fintune_test/mlp_2p_gated_on_hid; ext_opt="-o neurons_in_preprocess True" # similar performance..
# would you try back to use raw scores?
PREFIX=ak8_rawscores_topVsQCD_manual_pre2.gated.neurons_in_preprocess; network_name=fintune_test/mlp_2p_gated; ext_opt="-o neurons_in_preprocess True"
PREFIX=ak8_rawscores_topVsQCD_manual_pre2.gated_presoftmax.neurons_in_preprocess; network_name=fintune_test/mlp_2p_gated_forrawscores; ext_opt="-o neurons_in_preprocess True"
PREFIX=ak8_rawscores_topVsQCD_manual_pre2.gated_presoftmax.neurons_in_preprocess.skip_mlp; network_name=fintune_test/mlp_2p_gated_forrawscores; ext_opt="-o neurons_in_preprocess True -o skip_mlp True"
PREFIX=ak8_hidneurons_topVsQCD_manual_pre2.gated.neurons_in_preprocess.ly128; network_name=fintune_test/mlp_2p_gated; ext_opt="-o neurons_in_preprocess True -o layer_params 128,"
# 11.07 try
PREFIX=ak8_hidneurons_topVsQCD_manual_pre2.gated.neurons_in_preprocess.mlp_std; network_name=fintune_test/mlp_2p_gated; ext_opt="-o neurons_in_preprocess True -o no_last_relu True"
PREFIX=ak8_rawscores_topVsQCD_manual_pre2.gated_presoftmax.neurons_in_preprocess.skip_mlp.mlp_std; network_name=fintune_test/mlp_2p_gated_forrawscores; ext_opt="-o neurons_in_preprocess True -o skip_mlp True -o no_last_relu True"
PREFIX=ak8_rawscores_topVsQCD_manual_pre2.gated_presoftmax.skip_mlp.mlp_std; network_name=fintune_test/mlp_2p_gated_forrawscores; ext_opt="-o skip_mlp True -o no_last_relu True"

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 20 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}


## fine-tunings study #2
# adding H->bc scores

NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_addhbc ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

# well... why the performance get even worse..?
NGPUS=4
PREFIX=ak8_MD_vminclv2_pre2_origrecast ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

# get reweighting factors
NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

# formal training
NGPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--data-test \
'hww:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
'xbc:'$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*0.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# retry particlenet, but only with classification head
NGPUS=4
PREFIX=ak8_MD_vminclv2_fin_finetune_hbc_manual.zeus ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'xbc:'$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*0.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# let's inference Huilin's model!! (11.09)
PREFIX=ak8_MD_ParT_HQ ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger.py \
--model-prefix /home/olympus/hqu/work/nn/particle_transformer_CMS/particle_transformer/training/CMSAK8/MD/ParT/20220426-084550_example_ParticleTransformerTagger_ranger_lr0.002_batch512/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


## 22.11.07 try finetuning ParT!!
NGPUS=4
PREFIX=ak8_MD_vminclv2ParT_fin_finetune_hbc ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

# first get the reweighting factor
NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp --batch-size 512 --start-lr 1e-3 --num-epochs 50 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

# ok let's tranfer to farm221
--use-amp --batch-size 512 --start-lr 1e-4 --num-epochs 30 --optimizer ranger \
--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler one-cycle --load-model-weights model/ak8_MD_vminclv2ParT_manual_fixwrap/net_best_epoch_state.pt \
--gpus 0 --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'xbc:'$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*0.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/pre_train.log

# meanwhile we'll play with finetuning
NGPUS=1
GPU=0
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_hidneurons_hbc_manual; network_name=mlp_std; ext_opt="-o layer_params ''"
PREFIX=ak8_hidneurons_hbc_manual.lr1e-2; network_name=mlp_std; ext_opt="-o layer_params ''"
PREFIX=ak8_hidneurons_hbc_manual.ly128; network_name=mlp_std; ext_opt="-o layer_params 128,"
PREFIX=ak8_rawscores_hbc_manual; network_name=mlp_std; ext_opt="-o layer_params '' -o prep_log True"
PREFIX=ak8_rawscores_hbc_manual.ly128; network_name=mlp_std; ext_opt="-o layer_params 128, -o prep_log True"
PREFIX=ak8_rawscores_hqqonly_hbc_manual; network_name=mlp_std; ext_opt="-o layer_params '' -o prep_log True -o prep_norm True"
PREFIX=ak8_rawscores_hqqonly_hbc_manual.lr1e-2; network_name=mlp_std; ext_opt="-o layer_params '' -o prep_log True -o prep_norm True"

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_hbc/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'xbc:'$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*0.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 20 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}

# now add hqq!! (finally)
PREFIX=ak8_hidneurons_hbc_manual; network_name=mlp_std; ext_opt="-o layer_params ''"
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_hbc/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU \
--data-test \
'hqq:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToBulkGravitonToHHTo4QTau_M-1000_narrow/*/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


################################################################
# a long gap

# 22.12.13 finetune the HWW mass

NGPUS=1
GPU=0
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_hidneurons_gghww_manual;  network_name=fintune_test/mlp_2p_gated_regression; ext_opt="-o neurons_in_preprocess True -o no_last_relu True" # overlapping dir with diff LR. Last one is 0.04
PREFIX=ak8_hidneurons_gghww_addFullInvMass.lr2e-2;  network_name=fintune_test/mlp_2p_gated_regression; ext_opt="-o neurons_in_preprocess True -o no_last_relu True"
PREFIX=ak8_hidneurons_gghww_noInvMass.lr2e-2;  network_name=fintune_test/mlp_2p_gated_regression; ext_opt="-o neurons_in_preprocess True -o no_last_relu True"
PREFIX=ak8_hidneurons_gghww_dummyMET.lr2e-2;  network_name=fintune_test/mlp_2p_gated_regression; ext_opt="-o neurons_in_preprocess True -o no_last_relu True"

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_gghww/${PREFIX%%.*}.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode regression --load-model-weights finetune_gghww_custom \
--batch-size 512 --start-lr 2e-2 --num-epochs 30 --optimizer ranger --lr-scheduler one-cycle ${ext_opt} \
--gpus $GPU --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass/01/*.root' \
--data-test \
'gghww:'$DATAPATH'/20221023_ak8_UL17_v6/GluGluToH1JetToWW_HEFT_MH-15to250_JHUVariableWMass/01/*.root' \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 20 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train_{auto}.log --tensorboard _${PREFIX}_{auto}
''
# infer ttbar samples
NGPUS=1
GPU=0
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_hidneurons_gghww_addFullInvMass.lr2e-2;  network_name=fintune_test/mlp_2p_gated_regression; ext_opt="-o neurons_in_preprocess True -o no_last_relu True"

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_gghww/${PREFIX%%.*}_infer.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode regression --predict ${ext_opt} \
--batch-size 512 --gpus $GPU \
--data-test \
'ofcttbarsemilep:'$DATAPATH'/20221023_ak8_UL17_v6/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
'ofcttbarfulllep:'$DATAPATH'/20221023_ak8_UL17_v6/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# infer the Z'->ttbar sample (but it's for other studies. the MET all assigned zero...)
NGPUS=1
GPU=0
DATAPATH=/mldata/licq/deepjetak8
PREFIX=ak8_hidneurons_gghww_addFullInvMass.lr2e-2;  network_name=fintune_test/mlp_2p_gated_regression; ext_opt="-o neurons_in_preprocess True -o no_last_relu True"

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/finetune_gghww/${PREFIX%%.*}_inferttbarfull.yaml

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode regression --predict ${ext_opt} \
--batch-size 512 --gpus $GPU \
--data-test \
'ttbarfull:'$DATAPATH'/20221023_ak8_UL17_v6/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/$network_name.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


# re-infer Z'->ttbar sample with more events...
GPU=0
PREFIX=ak8_MD_vminclv2ParT_manual_fixwrap ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_inferttbarfull.yaml
DATAPATH=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'ttbarfull:'$DATAPATH'/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# 23.04.15 train PNet with regression only..
NGPUS=2
PREFIX=ak8_MD_vminclv2_pre2_regonly_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode regression \
--batch-size 512 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_mass_regression.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

### then inference the ofcttbar samples for both cls-only and reg-only
GPU=1
PREFIX=ak8_MD_vminclv2_pre2_regonly_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_inferofcttbarv6.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --train-mode regression --predict \
--batch-size 512 --gpus $GPU \
--data-test \
'ofcttbarsemilep:'$DATAPATHIFR'/20221023_ak8_UL17_v6/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
'ofcttbarfulllep:'$DATAPATHIFR'/20221023_ak8_UL17_v6/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv_mass_regression.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

GPU=1
PREFIX=ak8_MD_vminclv2_pre2_noreg_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}_inferofcttbarv6.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--batch-size 512 --gpus $GPU \
--data-test \
'ofcttbarsemilep:'$DATAPATHIFR'/20221023_ak8_UL17_v6/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
'ofcttbarfulllep:'$DATAPATHIFR'/20221023_ak8_UL17_v6/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/01/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root


# 23.04.22
NGPUS=2
PREFIX=ak8_MD_vminclv2_pre2 ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
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
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net

PREFIX=ak8_MD_vminclv2ParT_manual_fixwrap ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/mldata/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--batch-size 256 --start-lr 5e-3 --num-epochs 30 --optimizer ranger \
--gpus 0 --data-train \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net