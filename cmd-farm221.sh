# also test training on farm221!! (exciting)
NGPUS=1
PREFIX=ak8_MD_vmincl_pre3 ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

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
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net

# now ParT
# double the bs
NGPUS=1
PREFIX=ak8_MD_vminclParT_test ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp --batch-size 1024 --start-lr 2e-3 --num-epochs 30 --optimizer ranger \
--gpus 1 --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 10 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net

# formal training
NGPUS=4
PREFIX=ak8_MD_vminclParT_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp --batch-size 1024 --start-lr 2e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}


## formal training for ParT cls+reg
NGPUS=4
PREFIX=ak8_MD_vminclParT_addreg_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp --batch-size 1024 --start-lr 2e-3 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}

# prediction
GPU=0
PREFIX=ak8_MD_vminclParT_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--use-amp --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'xqq:'$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'hww:'$DATAPATH'/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:'$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# prediction for cls+reg
GPU=0
PREFIX=ak8_MD_vminclParT_addreg_manual
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp --batch-size 512 --num-workers 3 \
--gpus ${GPU} --data-test \
'xqq:'$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'hww:'$DATAPATH'/20220625_ak8_UL17_v4/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*/*.root' \
'qcd:'$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'ttbar:'$DATAPATH'/20220625_ak8_UL17_v4/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*/*.root' \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## official training of the last ParT cls+reg !!
NGPUS=4
PREFIX=ak8_MD_vminclv2ParT_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp --batch-size 1024 --start-lr 2e-3 --num-epochs 50 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}


## OK!!! Now we just fix the padding mode.... Do the last training
NGPUS=4
PREFIX=ak8_MD_vminclv2ParT_manual_fixwrap ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/incl/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o loss_gamma 0.05 \
--use-amp --batch-size 1024 --start-lr 2e-3 --num-epochs 50 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20220625_ak8_UL17_v4/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20220625_ak8_UL17_v4/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*/*.root' \
$DATAPATH'/20220915_ak8_UL17_v5/Spin0ToTT_VariableMass_W*_MX-600to6000_MH-15to250/*/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX}


##################
## 22.11.07 finetune 2 here ## This is a failure attempt..!
NGPUS=4
PREFIX=ak8_MD_vminclv2ParT_fin_finetune_hbc ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX//./_}.yaml
DATAPATH=/data/licq/samples/deepjetak8

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--use-amp --batch-size 1024 --start-lr 8e-4 --num-epochs 30 --optimizer ranger \
--optimizer-option lr_mult "(\"fc.*\",50)" --lr-scheduler one-cycle --load-model-weights model/ak8_MD_vminclv2ParT_manual_fixwrap/net_best_epoch_state.pt \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'xbc:'$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*0.root' \
--samples-per-epoch $((1000 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformerTagger_finetune.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

# try particlenet here # halt!!
NGPUS=4
PREFIX=ak8_MD_vminclv2_fin_finetune_hbc_manual ## needs modification
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv6/${PREFIX%%.*}.yaml
DATAPATH=/data/licq/samples/deepjetak8

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $HOME/hww/incl-train/weaver-core/weaver/train.py \
--batch-size 1024 --start-lr 1e-2 --num-epochs 30 --optimizer ranger \
--backend nccl --data-train \
$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*.root' \
$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*.root' \
--data-test \
'qcd:'$DATAPATH'/20221023_ak8_UL17_v6/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/01/*.root' \
'xqq:'$DATAPATH'/20221023_ak8_UL17_v6/BulkGravitonToHHTo4QTau_MX-600to6000_MH-15to250/*/*0.root' \
'xbc:'$DATAPATH'/20221023_ak8_UL17_v6/ChargedHiggs_HplusToBC_HminusToBC/*/*0.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 3 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/particle_net_pf_sv.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--log $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root