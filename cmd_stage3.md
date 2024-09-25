# 23.11.24 GloParT-stage3

## split regression per class

PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_split_reg True -o loss_gamma 5 -o fc_params '[(1024,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.00666667 \
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

## FC to 2048
// change LR to 1e-3 in 26-the epoch
// a failure attempt!
PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr2e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_split_reg True -o loss_gamma 5 -o fc_params '[(2048,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.00666667 \
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

## prediction for FC2048

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict --gpus 1 \
--train-mode hybrid -o three_coll True -o loss_split_reg True -o loss_gamma 5 -o fc_params '[(2048,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 1e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.00666667 \
--data-test \
'xww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*0.root' \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
'ofcttbarfl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/*.root' \
'ofcttbarsl:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 8 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_epoch-26_state.pt \
--log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## Tuning the model
PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr5e-4; lr='5e-4'
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=0,1,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_split_reg True -o loss_gamma 5 -o fc_params '[(2048,0.1)]'  -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr $lr --num-epochs 50 --optimizer ranger --fetch-step 0.00666667 \
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


# 23.12.29 stage2.5 variable studies
## use part-lite // DEPRECATED

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.lite.gm5.ddp-bs768-lr6p75e-3
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

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

# 24.08.05 variable studies
// use all new utilities: --run-mode train-only, --data-split-group 4
// for normpair: use normpair data_config, and configure "-o use_pair_norm True -o pair_input_dim 6"

// NOTE: At the time of study, the nodewise input takes both un-norm and norm variables => This is known to worsen the performance!!

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_normpair_manual.useamp.large.gm5.ddp-bs256-lr2e-3
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_normpair_manual.useamp.large.gm5.ddp-bs256-lr2e-3.addnonscale # add back original part pt/e inputs

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.02 --data-split-group 4 \
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


// # only add node-wise norm feats; impl new normpair feats in model
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_addnorm_manual.useamp.large.gm5.ddp-bs256-lr2e-3.modelnormpair 
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_addnorm_manual.useamp.large.gm5.ddp-bs256-lr2e-3.modelnormpair.splitnum1 # not this one.. why not able to reproduce the old training?
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3.modelnormpair

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o use_pair_norm True -o pair_input_dim 6 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.02 --data-split-group 1 \
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

/// forgot to add mask to calculate jet mass. Should check the pt-onnx result first!

## ========== test onnx model =============
/// export 

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_addnorm_manual.useamp.large.gm5.ddp-bs256-lr2e-3.modelnormpair
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

/// for Run2 model, remember to replace einsum, and run with opset 11
// use export_embed = True

GPU=0
python $HOME/hww/incl-train/weaver-core/weaver/train.py \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o use_pair_norm True -o pair_input_dim 6 \
-o export_embed True \
--use-amp \
--data-config ${config} \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/net_epoch-8_state.pt \
--export-onnx $HOME/hww/incl-train/weaver-core/weaver/model/$PREFIX/model_embed.onnx

## test onnx sample
// use the same model!!!

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}_inferonnxmodel.yaml
NGPUS=1

python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o use_pair_norm True -o pair_input_dim 6 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 1 --in-memory \
--gpus 0 \
--data-test \
'onnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 1 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net_epoch-8_state.pt \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// ========== end test onnx model =============

## now debug...

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_addnorm_manual.useamp.large.gm5.ddp-bs256-lr2e-3.modelnormpair.TEST

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

python $HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 -o use_pair_norm True -o pair_input_dim 6 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.05 --data-split-group 1 \
--gpus 0 --data-train \
$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*00.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 2 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net

## reproduce previous one..

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3.REPRODUCE

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

NGPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS \
$HOME/hww/incl-train/weaver-core/weaver/train.py --run-mode train-only \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 --data-split-group 2 \
--backend nccl --data-train \
't_qcd:'$DATAPATH'/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
't_ttbar:'$DATAPATH'/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root' \
't_h2p:'$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root' \
't_hpm2p:'$DATAPATH'/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root' \
't_hww:'$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root' \
't_hwxwx:'$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root' \
't_hzz:'$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root' \
't_hzxzx:'$DATAPATH'/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root' \
--data-test \
'hww:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root' \
'higgs2p:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root' \
'qcd:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root' \
'ttbar:'$DATAPATHIFR'/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net --log-file $HOME/hww/incl-train/weaver-core/weaver/logs/${PREFIX}/train.log --tensorboard _${PREFIX} \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

## infer the new higgs sample
// note that, at some point all data yaml cards has changed their test_time_selection. 
// The following cut is removed! (fj_pt>200) & (fj_pt<2500) & (fj_sdmass>=20) & (fj_sdmass<260)

### official v2
PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3.REPRODUCE

config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/pubfs/licq/deepjetak8

### split reg (add -o loss_split_reg True -o fc_params '[(2048,0.1)]')

PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr5e-4
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
DATAPATH=/mldata/licq/deepjetak8
DATAPATHIFR=/data/bond/licq/deepjetak8

NGPUS=1
python $HOME/hww/incl-train/weaver-core/weaver/train.py --predict \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params '[(1024,0.1)]' -o embed_dims '[256,1024,256]' -o pair_embed_dims '[128,128,128]' -o num_heads 16 \
--use-amp --batch-size 768 --start-lr 6.75e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.008 \
--gpus 2 \
--data-test \
'higlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*.root' \
'highi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*.root' \
'hwwlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-8_narrow/*.root' \
'hwwhi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-20_narrow/*.root' \
'zhi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/BulkGravToZZToZhadZhad_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia/*.root' \
'zlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/BulkGravToZZToZhadZhad_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia/*.root' \
'whi:'$DATAPATH'/20230504_ak8_UL17_v8/infer/BulkGravToWWToWhadWhad_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia/*.root' \
'wlo:'$DATAPATH'/20230504_ak8_UL17_v8/infer/BulkGravToWWToWhadWhad_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia/*.root' \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config $HOME/hww/incl-train/weaver-core/weaver/networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix $HOME/hww/incl-train/weaver-core/weaver/model/${PREFIX}/net \
--predict-output $HOME/hww/incl-train/weaver-core/weaver/predict/$PREFIX/pred.root

// how the files are suffled:
 - group the files, each group has an order
 - shuffle the order in each group (DDP mode)
 - concatenate all group and send to load_next -> shuffled again!
 - take filelist[i::N]

# 24.08.08 Formal run. Only GloParT v2 w/ split_reg + pair_norm
// Now you have a script.

PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr5e-4.modelnormpair
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml

source scripts/train_GloParT.sh run 1,2,3 --batch-size 256 --start-lr 5e-4 -o loss_split_reg True -o fc_params '[(2048,0.1)]' -o use_pair_norm True -o pair_input_dim 6
source scripts/train_GloParT.sh run 0 --batch-size 256 --start-lr 5e-4 -o loss_split_reg True -o fc_params '[(2048,0.1)]' -o use_pair_norm True -o pair_input_dim 6 --samples-per-epoch-val $((2000 * 512)) --run-mode val
source scripts/train_GloParT.sh dryrun 3 --batch-size 256 --start-lr 5e-4 -o loss_split_reg True -o fc_params '[(2048,0.1)]' -o use_pair_norm True -o pair_input_dim 6 --samples-per-epoch-val $((2000 * 512)) --run-mode test --data-split-group 1 --fetch-step 0.1

## let's switch to the v3 model

PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr5e-4.v3init
PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs512-lr7e-4.v3init
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml

source scripts/train_GloParT.sh run 4,5,6 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 -o loss_split_reg True -o fc_params '[(2048,0.1)]' -o use_swiglu_config True -o use_pair_norm_config True --fetch-step 0.02 --data-split-group 7
source scripts/train_GloParT.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 -o loss_split_reg True -o fc_params '[(2048,0.1)]' -o use_swiglu_config True -o use_pair_norm_config True --samples-per-epoch-val $((2000 * 512)) --run-mode val,test
// this runs on farm221
//--log-file logs/TEST/train.log --tensorboard _TEST --data-train t_qcd:./datasets/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*0.root --in-memory

## [for documentation] special train/test configs
// for val, use --samples-per-epoch-val $((4000 * 512)) --data-split-group [x3]
// for test, use --data-split-group 1

# 24.08.10 SwiGLU with more layers

PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs512-lr7e-4.v3init.embed192_h12_block16
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
modelopts="-o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [192,768,192] -o pair_embed_dims [96,96,96] -o num_heads 12 -o num_layers 16 "

// tried: --num-workers 8 --fetch-step 0.01 --data-split-group 5. If on farm221, need to use larger fetch-step
// --num-workers 5 --fetch-step 0.025 --data-split-group 12

source scripts/train_GloParT.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts --num-workers 5 --fetch-step 0.025 --data-split-group 12 

source scripts/train_GloParT.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts --num-workers 8 --run-mode val
source scripts/train_GloParT.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts --num-workers 8 --run-mode val --load-epoch 40 --samples-per-epoch-val $((4000 * 512)) --data-split-group 9 # do val again with larger samples per batch
source scripts/train_GloParT.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts --num-workers 8 --run-mode test --data-split-group 1

### pt-onnx test
// to export onnx: add -o export_params {"num_cls":314, "concat_hid":True} --model-prefix [..] --export-onnx [..]
source scripts/train_GloParT.sh run 3 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts -o export_params '{"num_cls":314,"concat_hid":True}' --model-prefix model/$PREFIX/net_epoch-46_state.pt --export-onnx model/$PREFIX/model_embed.onnx


# 24.08.10 SwiGLU hparams tuning

// fewer heads & layers
PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs512-lr7e-4.v3init.embed192_h6_block12
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus/${PREFIX%%.*}.yaml
modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [192,768,192] -o pair_embed_dims [96,96,96] -o num_heads 6 -o num_layers 12 "
trainopts="--num-workers 5 --fetch-step 0.025 --data-split-group 12 " # farm221 training
valopts="--run-mode val --num-workers 8 "
testopts="--run-mode test --num-workers 8 --data-split-group 1 "


source scripts/train_GloParT.sh run 4,5,6 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

# 24.08.17 ext-mass tuning

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_addltphp_wmeasonly_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs256-lr5e-4.v3init
config=$HOME/hww/incl-train/weaver-core/weaver/data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 "
trainopts="--num-workers 8 --fetch-step 0.04 --data-split-group 36 " # farm221 training
valopts="--run-mode val --num-workers 10 "
testopts="--run-mode test --num-workers 8 --data-split-group 1 "

source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts --log-file logs/${PREFIX}/pre_train.log
// start training
source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 3 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts

## 24.08.20 ihep: change to (2048, 0.2) and remove pixel hits, and num-epochs = 80

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch80.v3init
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 80 "
trainopts="--num-workers 8 --fetch-step 0.04 --data-split-group 36 " # farm221 training
valopts="--run-mode val --num-workers 10 "
testopts="--run-mode test --num-workers 8 --data-split-group 1 "

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 384 --start-lr 6e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 3 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts

## farm221: also trying embed192_h12_block16

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs512-lr7e-4.nepoch80.v3init.embed192_h12_block12
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [192,768,192] -o pair_embed_dims [96,96,96] -o num_heads 12 -o num_layers 12 --num-epochs 80 "
trainopts="--num-workers 8 --fetch-step 0.04 --data-split-group 36 " # farm221 training
valopts="--run-mode val --num-workers 10 "
testopts="--run-mode test --num-workers 8 --data-split-group 1 "

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts

// a small test..
PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs512-lr7e-4.nepoch80.v3init.embed192_h12_block12.lowmassval
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml
modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [192,768,192] -o pair_embed_dims [96,96,96] -o num_heads 12 -o num_layers 12 --num-epochs 80 "
trainopts="--num-workers 8 --fetch-step 0.04 --data-split-group 36 " # farm221 training
valopts="--run-mode val --num-workers 10 "
source scripts/train_GloParT.sh run 3 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
// better val acc, but still large fluctuation. could be due to uneven splitting?


source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 256 --start-lr 7e-4 $modelopts $trainopts --fetch-step 1 --data-split-group 100 --num-workers 10

## farm221: fix split

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs512-lr7e-4.nepoch80.v3fixsplit2.embed192_h12_block12
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [192,768,192] -o pair_embed_dims [96,96,96] -o num_heads 12 -o num_layers 12 --num-epochs 80 "
trainopts="--num-workers 8 --fetch-step 1. --data-split-group 500 " 
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

##!!!
source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 3 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT_extmass.sh dryrun 3 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

## again... ihep (this is an important test):

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch80.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 80 "
trainopts="--num-workers 8 --fetch-step 1. --data-split-group 500 " # on ihep we should use 500. larger memory sometimes causes crashs..
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 384 --start-lr 6e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

### coming back 08.28: extend to 120 epochs, (relax dropout to 0.1)?

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch120_ext.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 120 "
trainopts="--num-workers 8 --fetch-step 1. --data-split-group 400 "
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 384 --start-lr 6e-4 $modelopts $trainopts --load-epoch 55
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts --load-epoch 55
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

### or we maintain dropout=0.2 -> on farm221

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch120_ext-dp0p2.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 120 "
trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # allow larger memory
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts --load-epoch 55
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts --load-epoch 55
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

// why?? there is overtraining again.. does changing dropout=0.1 in the middle helps...?

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch120_ext-dp0p1re.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml
modelopts="-o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 120 "

// change to --num-epochs 100 ; finalize the training
### the temporary best model for benchmarking

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch100.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml
modelopts="-o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # allow larger memory
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts --load-epoch 69
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts --load-epoch 69
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

### test onnx export

// to export onnx: add -o export_params {"num_cls":314, "concat_hid":True} --model-prefix [..] --export-onnx [..]

source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts -o export_params '{"num_cls":313,"concat_hid":True}' --model-prefix model/$PREFIX/net_best_epoch_state.pt --export-onnx model/$PREFIX/model_embed.onnx

// test samples from dnntuple onnx inference

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch100.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}_inferonnxmodel.yaml

extopts="--data-test onnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root "
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts $extopts



## we should split the QCD sample in case of using --data-split-group

jobid=8065357
jobid=8065360 # for infer
jobid=7216497 # for infer_UL17
qcddir=QCD_Pt_170to300_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 0 49`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_300to470_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 50 99`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_470to600_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 100 149`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_600to800_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 150 199`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_800to1000_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 200 249`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 250 299`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 300 349`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 350 399`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 400 449`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 450 474`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;

// record: hadd commands
hadd pred_higlo.root pred_higlo_part*
hadd pred_highi.root pred_highi_part*

// again... with v3fixsplit2

## new on farm221: smaller LR (not working..)


PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs512-lr1p5e-4.nepoch80.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o fc_params [(2048,0.2)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 80 "
trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # allow larger memory
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 2,3,4 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 1.5e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 1.5e-4 $modelopts $valopts

## ========== new save_root impl. =============

PREFIX=ak8_MD_inclv8_part_addltphp_wmeasonly_manual.useamp.large.gm5.ddp-bs256-lr2e-3  ## original stage-2
config=./data_new/inclv7plus/${PREFIX%%.*}.yaml
modelopts=""

PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr5e-4  ## +splitreg
config=./data_new/inclv7plus/${PREFIX%%.*}.yaml
modelopts="-o loss_split_reg True -o fc_params [(2048,0.1)] "

PREFIX=ak8_MD_inclv8_part_splitreg_addltphp_wmeasonly_manual.useamp.large_fc2048.gm5.ddp-bs256-lr5e-4.modelnormpair
config=./data_new/inclv7plus/${PREFIX%%.*}.yaml
modelopts="-o loss_split_reg True -o fc_params [(2048,0.1)] -o use_pair_norm True -o pair_input_dim 6 "

testopts="--run-mode test --num-workers 8 --data-split-group 1 "
extopts="--network-config networks/example_ParticleTransformer2023Tagger_hybrid_saveroot.py --predict-output predict/$PREFIX/new/pred.root " ## to distinguish from original save_root
source scripts/train_GloParT_extmass.sh dryrun 2 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts $extopts

=================================

## re-tune gamma=1

// more dataset -> should be ok to use dropout=0.1?
// seems working. But rerun with expanded inputs
// eventually not used

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048dp0p1.gm1.ddp-bs512-lr3e-3.nepoch80.v3fixsplit2 # LR too small

PREFIX=ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048dp0p1.gm1.ddp-bs512-lr6e-4.nepoch120.v3fixsplit2 # observed over-training since ~60 epochs. should be related to r=0.1..
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o loss_split_reg True -o loss_gamma 1 -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 120 "
trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # allow larger memory
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 1,2,3 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 6e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger.py --batch-size 512 --start-lr 6e-4 $modelopts $valopts
<!-- 
# Go to v10

# 24.08.29 print a dummy model
// to export onnx: add -o export_params {"num_cls":314, "concat_hid":True} --model-prefix [..] --export-onnx [..]
// but here we just export dummy model

PREFIX=ak8_MD_inclv10_manual.dummy
config=./data_new/inclv10/${PREFIX%%.*}.yaml

source scripts/train_GloParT_extmass.sh dryrun 3 --network-config networks/dummy/example_GloParT3_dummy.py --batch-size 512 --model-prefix model/$PREFIX/dummy_model.pt --export-onnx model/$PREFIX/model.onnx
 -->

## 24.08.29 switch to standard weaver-core impl.

// use _std yaml cards (without specifying split_per_cls) and _std model (use customized train/eval functions)

PREFIX=ak8_MD_inclv8std_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch100.v3fixsplit2
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml
modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # allow larger memory
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

extopts="--data-train t_h2p:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*00.root --samples-per-epoch 1024 --samples-per-epoch-val 1024 --run-mode train,val"

source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_std.py --batch-size 256 --start-lr 5e-4 $modelopts $trainopts $extopts


## 24.09.03 with standard input (same as stage-2)
// change model: lighter pair_embed in change of more layers

// this must be provided now for test step (v3pre: 313 class with no H_bs)
label_cls_nodes_v3pre="['label_Top_bWcs','label_Top_bWqq','label_Top_bWc','label_Top_bWs','label_Top_bWq','label_Top_bWev','label_Top_bWmv','label_Top_bWtauev','label_Top_bWtaumv','label_Top_bWtauhv','label_Top_Wcs','label_Top_Wqq','label_Top_Wev','label_Top_Wmv','label_Top_Wtauev','label_Top_Wtaumv','label_Top_Wtauhv','label_H_bb','label_H_cc','label_H_ss','label_H_qq','label_H_bc','label_H_cs','label_H_gg','label_H_ee','label_H_mm','label_H_tauhtaue','label_H_tauhtaum','label_H_tauhtauh','label_H_WW_cscs','label_H_WW_csqq','label_H_WW_qqqq','label_H_WW_csc','label_H_WW_css','label_H_WW_csq','label_H_WW_qqc','label_H_WW_qqs','label_H_WW_qqq','label_H_WW_csev','label_H_WW_qqev','label_H_WW_csmv','label_H_WW_qqmv','label_H_WW_cstauev','label_H_WW_qqtauev','label_H_WW_cstaumv','label_H_WW_qqtaumv','label_H_WW_cstauhv','label_H_WW_qqtauhv','label_H_WxWx_cscs','label_H_WxWx_csqq','label_H_WxWx_qqqq','label_H_WxWx_csc','label_H_WxWx_css','label_H_WxWx_csq','label_H_WxWx_qqc','label_H_WxWx_qqs','label_H_WxWx_qqq','label_H_WxWx_csev','label_H_WxWx_qqev','label_H_WxWx_csmv','label_H_WxWx_qqmv','label_H_WxWx_cstauev','label_H_WxWx_qqtauev','label_H_WxWx_cstaumv','label_H_WxWx_qqtaumv','label_H_WxWx_cstauhv','label_H_WxWx_qqtauhv','label_H_WxWxStar_cscs','label_H_WxWxStar_csqq','label_H_WxWxStar_qqqq','label_H_WxWxStar_csc','label_H_WxWxStar_css','label_H_WxWxStar_csq','label_H_WxWxStar_qqc','label_H_WxWxStar_qqs','label_H_WxWxStar_qqq','label_H_WxWxStar_csev','label_H_WxWxStar_qqev','label_H_WxWxStar_csmv','label_H_WxWxStar_qqmv','label_H_WxWxStar_cstauev','label_H_WxWxStar_qqtauev','label_H_WxWxStar_cstaumv','label_H_WxWxStar_qqtaumv','label_H_WxWxStar_cstauhv','label_H_WxWxStar_qqtauhv','label_H_ZZ_bbbb','label_H_ZZ_bbcc','label_H_ZZ_bbss','label_H_ZZ_bbqq','label_H_ZZ_cccc','label_H_ZZ_ccss','label_H_ZZ_ccqq','label_H_ZZ_ssss','label_H_ZZ_ssqq','label_H_ZZ_qqqq','label_H_ZZ_bbb','label_H_ZZ_bbc','label_H_ZZ_bbs','label_H_ZZ_bbq','label_H_ZZ_ccb','label_H_ZZ_ccc','label_H_ZZ_ccs','label_H_ZZ_ccq','label_H_ZZ_ssb','label_H_ZZ_ssc','label_H_ZZ_sss','label_H_ZZ_ssq','label_H_ZZ_qqb','label_H_ZZ_qqc','label_H_ZZ_qqs','label_H_ZZ_qqq','label_H_ZZ_bbee','label_H_ZZ_bbmm','label_H_ZZ_bbe','label_H_ZZ_bbm','label_H_ZZ_bee','label_H_ZZ_bmm','label_H_ZZ_bbtauhtaue','label_H_ZZ_bbtauhtaum','label_H_ZZ_bbtauhtauh','label_H_ZZ_btauhtaue','label_H_ZZ_btauhtaum','label_H_ZZ_btauhtauh','label_H_ZZ_ccee','label_H_ZZ_ccmm','label_H_ZZ_cce','label_H_ZZ_ccm','label_H_ZZ_cee','label_H_ZZ_cmm','label_H_ZZ_cctauhtaue','label_H_ZZ_cctauhtaum','label_H_ZZ_cctauhtauh','label_H_ZZ_ctauhtaue','label_H_ZZ_ctauhtaum','label_H_ZZ_ctauhtauh','label_H_ZZ_ssee','label_H_ZZ_ssmm','label_H_ZZ_sse','label_H_ZZ_ssm','label_H_ZZ_see','label_H_ZZ_smm','label_H_ZZ_sstauhtaue','label_H_ZZ_sstauhtaum','label_H_ZZ_sstauhtauh','label_H_ZZ_stauhtaue','label_H_ZZ_stauhtaum','label_H_ZZ_stauhtauh','label_H_ZZ_qqee','label_H_ZZ_qqmm','label_H_ZZ_qqe','label_H_ZZ_qqm','label_H_ZZ_qee','label_H_ZZ_qmm','label_H_ZZ_qqtauhtaue','label_H_ZZ_qqtauhtaum','label_H_ZZ_qqtauhtauh','label_H_ZZ_qtauhtaue','label_H_ZZ_qtauhtaum','label_H_ZZ_qtauhtauh','label_H_ZxZx_bbbb','label_H_ZxZx_bbcc','label_H_ZxZx_bbss','label_H_ZxZx_bbqq','label_H_ZxZx_cccc','label_H_ZxZx_ccss','label_H_ZxZx_ccqq','label_H_ZxZx_ssss','label_H_ZxZx_ssqq','label_H_ZxZx_qqqq','label_H_ZxZx_bbb','label_H_ZxZx_bbc','label_H_ZxZx_bbs','label_H_ZxZx_bbq','label_H_ZxZx_ccb','label_H_ZxZx_ccc','label_H_ZxZx_ccs','label_H_ZxZx_ccq','label_H_ZxZx_ssb','label_H_ZxZx_ssc','label_H_ZxZx_sss','label_H_ZxZx_ssq','label_H_ZxZx_qqb','label_H_ZxZx_qqc','label_H_ZxZx_qqs','label_H_ZxZx_qqq','label_H_ZxZx_bbee','label_H_ZxZx_bbmm','label_H_ZxZx_bbe','label_H_ZxZx_bbm','label_H_ZxZx_bee','label_H_ZxZx_bmm','label_H_ZxZx_bbtauhtaue','label_H_ZxZx_bbtauhtaum','label_H_ZxZx_bbtauhtauh','label_H_ZxZx_btauhtaue','label_H_ZxZx_btauhtaum','label_H_ZxZx_btauhtauh','label_H_ZxZx_ccee','label_H_ZxZx_ccmm','label_H_ZxZx_cce','label_H_ZxZx_ccm','label_H_ZxZx_cee','label_H_ZxZx_cmm','label_H_ZxZx_cctauhtaue','label_H_ZxZx_cctauhtaum','label_H_ZxZx_cctauhtauh','label_H_ZxZx_ctauhtaue','label_H_ZxZx_ctauhtaum','label_H_ZxZx_ctauhtauh','label_H_ZxZx_ssee','label_H_ZxZx_ssmm','label_H_ZxZx_sse','label_H_ZxZx_ssm','label_H_ZxZx_see','label_H_ZxZx_smm','label_H_ZxZx_sstauhtaue','label_H_ZxZx_sstauhtaum','label_H_ZxZx_sstauhtauh','label_H_ZxZx_stauhtaue','label_H_ZxZx_stauhtaum','label_H_ZxZx_stauhtauh','label_H_ZxZx_qqee','label_H_ZxZx_qqmm','label_H_ZxZx_qqe','label_H_ZxZx_qqm','label_H_ZxZx_qee','label_H_ZxZx_qmm','label_H_ZxZx_qqtauhtaue','label_H_ZxZx_qqtauhtaum','label_H_ZxZx_qqtauhtauh','label_H_ZxZx_qtauhtaue','label_H_ZxZx_qtauhtaum','label_H_ZxZx_qtauhtauh','label_H_ZxZxStar_bbbb','label_H_ZxZxStar_bbcc','label_H_ZxZxStar_bbss','label_H_ZxZxStar_bbqq','label_H_ZxZxStar_cccc','label_H_ZxZxStar_ccss','label_H_ZxZxStar_ccqq','label_H_ZxZxStar_ssss','label_H_ZxZxStar_ssqq','label_H_ZxZxStar_qqqq','label_H_ZxZxStar_bbb','label_H_ZxZxStar_bbc','label_H_ZxZxStar_bbs','label_H_ZxZxStar_bbq','label_H_ZxZxStar_ccb','label_H_ZxZxStar_ccc','label_H_ZxZxStar_ccs','label_H_ZxZxStar_ccq','label_H_ZxZxStar_ssb','label_H_ZxZxStar_ssc','label_H_ZxZxStar_sss','label_H_ZxZxStar_ssq','label_H_ZxZxStar_qqb','label_H_ZxZxStar_qqc','label_H_ZxZxStar_qqs','label_H_ZxZxStar_qqq','label_H_ZxZxStar_bbee','label_H_ZxZxStar_bbmm','label_H_ZxZxStar_bbe','label_H_ZxZxStar_bbm','label_H_ZxZxStar_bee','label_H_ZxZxStar_bmm','label_H_ZxZxStar_bbtauhtaue','label_H_ZxZxStar_bbtauhtaum','label_H_ZxZxStar_bbtauhtauh','label_H_ZxZxStar_btauhtaue','label_H_ZxZxStar_btauhtaum','label_H_ZxZxStar_btauhtauh','label_H_ZxZxStar_ccee','label_H_ZxZxStar_ccmm','label_H_ZxZxStar_cce','label_H_ZxZxStar_ccm','label_H_ZxZxStar_cee','label_H_ZxZxStar_cmm','label_H_ZxZxStar_cctauhtaue','label_H_ZxZxStar_cctauhtaum','label_H_ZxZxStar_cctauhtauh','label_H_ZxZxStar_ctauhtaue','label_H_ZxZxStar_ctauhtaum','label_H_ZxZxStar_ctauhtauh','label_H_ZxZxStar_ssee','label_H_ZxZxStar_ssmm','label_H_ZxZxStar_sse','label_H_ZxZxStar_ssm','label_H_ZxZxStar_see','label_H_ZxZxStar_smm','label_H_ZxZxStar_sstauhtaue','label_H_ZxZxStar_sstauhtaum','label_H_ZxZxStar_sstauhtauh','label_H_ZxZxStar_stauhtaue','label_H_ZxZxStar_stauhtaum','label_H_ZxZxStar_stauhtauh','label_H_ZxZxStar_qqee','label_H_ZxZxStar_qqmm','label_H_ZxZxStar_qqe','label_H_ZxZxStar_qqm','label_H_ZxZxStar_qee','label_H_ZxZxStar_qmm','label_H_ZxZxStar_qqtauhtaue','label_H_ZxZxStar_qqtauhtaum','label_H_ZxZxStar_qqtauhtauh','label_H_ZxZxStar_qtauhtaue','label_H_ZxZxStar_qtauhtaum','label_H_ZxZxStar_qtauhtauh','label_QCD_bb','label_QCD_cc','label_QCD_b','label_QCD_c','label_QCD_others']"

PREFIX=ak8_MD_inclv8std_rmhbs_manual.useamp.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100 ## best model for now -> labeled beta 1
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 --num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 5,6,7 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

// continue on ihep at epoch 61
trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts

// new evaluation: make ROC curve

valextopts="-o eval_kw {'roc_kw':{'comp_list':[('Xbb','QCD'),('Xcc','QCD'),('Xcc','Xbb')],'label_inds_map':{'Xbb':[17],'Xcc':[18],'QCD':[308,309,310,311,312]}}} "

source scripts/train_GloParT_extmass.sh run 2 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts $valextopts --tensorboard _${PREFIX}.withroc

### test onnx model (v3 beta 1)

PREFIX=ak8_MD_inclv8std_rmhbs_manual.useamp.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100 ## best model for now -> labeled beta 1
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}_inferonnxmodel.yaml

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 --num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

extopts="--data-test onnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root "

source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts $extopts


## 24.09.10 quick test on nblocks=12, FC dim=3072 (halted at 25 epochs)

PREFIX=ak8_MD_inclv8std_rmhbs_manual.useamp.large_fc3072.pemb64_block12.gm5.ddp-bs512-lr7e-4.nepoch100 ## best model for now -> labeled beta 1
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(3072,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 12 --num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts

## 24.09.12 composed regression

// YAML card changed to "compreg"
// node +2, -o loss_composed_split_reg [True,False] # True for "res", False for "vispart"

PREFIX=ak8_MD_inclv8std_compreg_rmhbs_manual.useamp.large_fc2048.pemb64_block12.gm5.ddp-bs512-lr7e-4.nepoch100 ## best model for now -> labeled beta 1
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 628 -o num_cls_nodes 313 -o loss_composed_split_reg [True,False] -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 12 --num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

// for test
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 256 --start-lr 7e-4 $modelopts $trainopts --data-train t_h2p:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*00.root --samples-per-epoch 512 --samples-per-epoch-val 512 --run-mode train,val


source scripts/train_GloParT_extmass.sh run 0,1,2 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts


========================================================================================================
## Mid-term stage-3: using 100-epoch version to do non-MD transfer learning

### init attempt

PREFIX=ak8_MD_inclv8std_nonmd_manual.test2
config=./data_new/inclv7plus_nonmd/${PREFIX%%.*}.yaml

load_model="./model/ak8_MD_inclv8_part_splitreg_extmass_rmhbs_manual.useamp.large_fc2048.gm5.ddp-bs384-lr6e-4.nepoch100.v3fixsplit2/net_best_epoch_state.pt"
label_cls_nodes="['label_Top_bWcs','label_Top_bWqq','label_Top_bWc','label_Top_bWs','label_Top_bWq','label_Top_bWev','label_Top_bWmv','label_Top_bWtauev','label_Top_bWtaumv','label_Top_bWtauhv','label_Top_Wcs','label_Top_Wqq','label_Top_Wev','label_Top_Wmv','label_Top_Wtauev','label_Top_Wtaumv','label_Top_Wtauhv','label_W_cs','label_W_qq','label_Z_bb','label_Z_cc','label_Z_ss','label_Z_qq','label_QCD_bb','label_QCD_cc','label_QCD_b','label_QCD_c','label_QCD_others']"

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 --num-epochs 100 "
modelftopts="-o num_nodes 28 -o num_cls_nodes 28 -o label_cls_nodes ${label_cls_nodes} -o loss_gamma 0 -o use_external_fc True -o fc_params [(256,0),(256,0)] --num-epochs 30 --load-model-weights ${load_model} --exclude-model-weights part\\.fc.* --freeze-model-weights (input_embeds|part\\.(embed|pair_embed|blocks|cls_token|cls_blocks|norm)).* " # the fine-tuned model setup; may override some modelopts

trainopts="--run-mode train,val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_nonMD.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 2e-2 $modelopts $modelftopts $trainopts
source scripts/train_GloParT_nonMD.sh dryrun 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 2e-2 $modelopts $modelftopts $testopts

### 24.09.14 for v3 beta 1

PREFIX=ak8_MD_inclv8std_nonmd_withenergy_manual.origmodel.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv7plus_nonmd/${PREFIX%%.*}.yaml

///========== should always load this ============
load_model="./model/ak8_MD_inclv8std_rmhbs_manual.useamp.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100/net_best_epoch_state.pt" # beta 1's config
label_cls_nodes="['label_Top_bWcs','label_Top_bWqq','label_Top_bWc','label_Top_bWs','label_Top_bWq','label_Top_bWev','label_Top_bWmv','label_Top_bWtauev','label_Top_bWtaumv','label_Top_bWtauhv','label_Top_Wcs','label_Top_Wqq','label_Top_Wev','label_Top_Wmv','label_Top_Wtauev','label_Top_Wtaumv','label_Top_Wtauhv','label_W_cs','label_W_qq','label_Z_bb','label_Z_cc','label_Z_ss','label_Z_qq','label_QCD_bb','label_QCD_cc','label_QCD_b','label_QCD_c','label_QCD_others']"

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 --num-epochs 100 " # beta 1's config
modelftopts="-o num_nodes 28 -o num_cls_nodes 28 -o label_cls_nodes ${label_cls_nodes} -o loss_gamma 0 -o use_external_fc True -o fc_params [(256,0),(256,0)] --num-epochs 30 --load-model-weights ${load_model} --exclude-model-weights part\\.fc.* --freeze-model-weights (input_embeds|part\\.(embed|pair_embed|blocks|cls_token|cls_blocks|norm)).* " # the fine-tuned model setup; may override some modelopts

trainopts="--run-mode train,val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_nonMD.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 2e-2 $modelopts $modelftopts $trainopts
source scripts/train_GloParT_nonMD.sh dryrun 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 2e-2 $modelopts $modelftopts $testopts

### 24.09.16 not using energy for now; LR/5

PREFIX=ak8_MD_inclv8std_nonmd_manual.fix.origmodel.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv7plus_nonmd/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 --num-epochs 100 " # beta 1's config
modelftopts="-o num_nodes 28 -o num_cls_nodes 28 -o label_cls_nodes ${label_cls_nodes} -o loss_gamma 0 -o use_external_fc True -o fc_params [(256,0),(256,0)] --num-epochs 30 --load-model-weights ${load_model} --exclude-model-weights part\\.fc.* --freeze-model-weights (input_embeds|part\\.(embed|pair_embed|blocks|cls_token|cls_blocks|norm)).* " # the fine-tuned model setup; may override some modelopts

trainopts="--run-mode train,val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_nonMD.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 4e-3 $modelopts $modelftopts $trainopts
source scripts/train_GloParT_nonMD.sh dryrun 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 2e-2 $modelopts $modelftopts $testopts


### Fix bug: the ParT is not entirely frozon -> BatchNorm has running stats..
// add -o freeze_part True

PREFIX=ak8_MD_inclv8std_nonmd_manual.freezepart.origmodel.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv7plus_nonmd/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 --num-epochs 100 " # beta 1's config
modelftopts="-o num_nodes 28 -o num_cls_nodes 28 -o label_cls_nodes ${label_cls_nodes} -o loss_gamma 0 -o use_external_fc True -o freeze_main_params True -o fc_params [(256,0),(256,0)] --num-epochs 30 --load-model-weights ${load_model} --exclude-model-weights part\\.fc.* --freeze-model-weights (input_embeds|part\\.(embed|pair_embed|blocks|cls_token|cls_blocks|norm)).* " # the fine-tuned model setup; may override some modelopts

trainopts="--run-mode train,val --num-workers 20 --fetch-step 1. --data-split-group 125 --samples-per-epoch-val $((1000 * 512))"
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_nonMD.sh run 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 4e-3 $modelopts $modelftopts $trainopts
source scripts/train_GloParT_nonMD.sh dryrun 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 2e-2 $modelopts $modelftopts $testopts


### export entire beta1 model
->go to partv3_export_model.ipynb

### test onnx model (unified model exports)

PREFIX=ak8_MD_inclv8std_nonmd_withenergy_manual.origmodel.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100
PREFIX=ak8_MD_inclv8std_nonmd_manual.fix.origmodel.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100 # remove fj_energy input for now
PREFIX=ak8_MD_inclv8std_nonmd_manual.freezepart.origmodel.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv7plus_nonmd/${PREFIX%%.*}_inferonnxmodel.yaml

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 --num-epochs 100 " # beta 1's config
modelftopts="-o num_nodes 28 -o num_cls_nodes 28 -o label_cls_nodes ${label_cls_nodes} -o loss_gamma 0 -o use_external_fc True -o freeze_main_params True -o fc_params [(256,0),(256,0)] --num-epochs 30 --load-model-weights ${load_model} --exclude-model-weights part\\.fc.* --freeze-model-weights (input_embeds|part\\.(embed|pair_embed|blocks|cls_token|cls_blocks|norm)).* " # the fine-tuned model setup; may override some modelopts

testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

extopts="--data-test onnxtest:/home/olympus/licq/hww/incl-train/weaver-core/weaver/output_numEvent100.root "

source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $modelftopts $testopts $extopts

========================================================================================================
## 24.09.12 Finally changing to PUPPIv18 ("Run3") samples (v10 std setup)

// script: train_GloParT_v3.sh, changed to v3 configs

// get sample weights

PREFIX=ak8_MD_inclv10init.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="--num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

source scripts/train_GloParT_v3.sh run cpu --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 256 --start-lr 7e-4 $modelopts $trainopts --log-file logs/${PREFIX}/pre_train.log

### formal training -> labeled beta 2

label_cls_nodes_v3beta2="['label_Top_bWcs','label_Top_bWqq','label_Top_bWc','label_Top_bWs','label_Top_bWq','label_Top_bWev','label_Top_bWmv','label_Top_bWtauev','label_Top_bWtaumv','label_Top_bWtauhv','label_Top_Wcs','label_Top_Wqq','label_Top_Wev','label_Top_Wmv','label_Top_Wtauev','label_Top_Wtaumv','label_Top_Wtauhv','label_H_bb','label_H_cc','label_H_ss','label_H_qq','label_H_bc','label_H_bs','label_H_cs','label_H_gg','label_H_aa','label_H_ee','label_H_mm','label_H_tauhtaue','label_H_tauhtaum','label_H_tauhtauh','label_H_WW_cscs','label_H_WW_csqq','label_H_WW_qqqq','label_H_WW_csc','label_H_WW_css','label_H_WW_csq','label_H_WW_qqc','label_H_WW_qqs','label_H_WW_qqq','label_H_WW_csev','label_H_WW_qqev','label_H_WW_csmv','label_H_WW_qqmv','label_H_WW_cstauev','label_H_WW_qqtauev','label_H_WW_cstaumv','label_H_WW_qqtaumv','label_H_WW_cstauhv','label_H_WW_qqtauhv','label_H_WxWx_cscs','label_H_WxWx_csqq','label_H_WxWx_qqqq','label_H_WxWx_csc','label_H_WxWx_css','label_H_WxWx_csq','label_H_WxWx_qqc','label_H_WxWx_qqs','label_H_WxWx_qqq','label_H_WxWx_csev','label_H_WxWx_qqev','label_H_WxWx_csmv','label_H_WxWx_qqmv','label_H_WxWx_cstauev','label_H_WxWx_qqtauev','label_H_WxWx_cstaumv','label_H_WxWx_qqtaumv','label_H_WxWx_cstauhv','label_H_WxWx_qqtauhv','label_H_WxWxStar_cscs','label_H_WxWxStar_csqq','label_H_WxWxStar_qqqq','label_H_WxWxStar_csc','label_H_WxWxStar_css','label_H_WxWxStar_csq','label_H_WxWxStar_qqc','label_H_WxWxStar_qqs','label_H_WxWxStar_qqq','label_H_WxWxStar_csev','label_H_WxWxStar_qqev','label_H_WxWxStar_csmv','label_H_WxWxStar_qqmv','label_H_WxWxStar_cstauev','label_H_WxWxStar_qqtauev','label_H_WxWxStar_cstaumv','label_H_WxWxStar_qqtaumv','label_H_WxWxStar_cstauhv','label_H_WxWxStar_qqtauhv','label_H_ZZ_bbbb','label_H_ZZ_bbcc','label_H_ZZ_bbss','label_H_ZZ_bbqq','label_H_ZZ_cccc','label_H_ZZ_ccss','label_H_ZZ_ccqq','label_H_ZZ_ssss','label_H_ZZ_ssqq','label_H_ZZ_qqqq','label_H_ZZ_bbb','label_H_ZZ_bbc','label_H_ZZ_bbs','label_H_ZZ_bbq','label_H_ZZ_ccb','label_H_ZZ_ccc','label_H_ZZ_ccs','label_H_ZZ_ccq','label_H_ZZ_ssb','label_H_ZZ_ssc','label_H_ZZ_sss','label_H_ZZ_ssq','label_H_ZZ_qqb','label_H_ZZ_qqc','label_H_ZZ_qqs','label_H_ZZ_qqq','label_H_ZZ_bbee','label_H_ZZ_bbmm','label_H_ZZ_bbe','label_H_ZZ_bbm','label_H_ZZ_bee','label_H_ZZ_bmm','label_H_ZZ_bbtauhtaue','label_H_ZZ_bbtauhtaum','label_H_ZZ_bbtauhtauh','label_H_ZZ_btauhtaue','label_H_ZZ_btauhtaum','label_H_ZZ_btauhtauh','label_H_ZZ_ccee','label_H_ZZ_ccmm','label_H_ZZ_cce','label_H_ZZ_ccm','label_H_ZZ_cee','label_H_ZZ_cmm','label_H_ZZ_cctauhtaue','label_H_ZZ_cctauhtaum','label_H_ZZ_cctauhtauh','label_H_ZZ_ctauhtaue','label_H_ZZ_ctauhtaum','label_H_ZZ_ctauhtauh','label_H_ZZ_ssee','label_H_ZZ_ssmm','label_H_ZZ_sse','label_H_ZZ_ssm','label_H_ZZ_see','label_H_ZZ_smm','label_H_ZZ_sstauhtaue','label_H_ZZ_sstauhtaum','label_H_ZZ_sstauhtauh','label_H_ZZ_stauhtaue','label_H_ZZ_stauhtaum','label_H_ZZ_stauhtauh','label_H_ZZ_qqee','label_H_ZZ_qqmm','label_H_ZZ_qqe','label_H_ZZ_qqm','label_H_ZZ_qee','label_H_ZZ_qmm','label_H_ZZ_qqtauhtaue','label_H_ZZ_qqtauhtaum','label_H_ZZ_qqtauhtauh','label_H_ZZ_qtauhtaue','label_H_ZZ_qtauhtaum','label_H_ZZ_qtauhtauh','label_H_ZxZx_bbbb','label_H_ZxZx_bbcc','label_H_ZxZx_bbss','label_H_ZxZx_bbqq','label_H_ZxZx_cccc','label_H_ZxZx_ccss','label_H_ZxZx_ccqq','label_H_ZxZx_ssss','label_H_ZxZx_ssqq','label_H_ZxZx_qqqq','label_H_ZxZx_bbb','label_H_ZxZx_bbc','label_H_ZxZx_bbs','label_H_ZxZx_bbq','label_H_ZxZx_ccb','label_H_ZxZx_ccc','label_H_ZxZx_ccs','label_H_ZxZx_ccq','label_H_ZxZx_ssb','label_H_ZxZx_ssc','label_H_ZxZx_sss','label_H_ZxZx_ssq','label_H_ZxZx_qqb','label_H_ZxZx_qqc','label_H_ZxZx_qqs','label_H_ZxZx_qqq','label_H_ZxZx_bbee','label_H_ZxZx_bbmm','label_H_ZxZx_bbe','label_H_ZxZx_bbm','label_H_ZxZx_bee','label_H_ZxZx_bmm','label_H_ZxZx_bbtauhtaue','label_H_ZxZx_bbtauhtaum','label_H_ZxZx_bbtauhtauh','label_H_ZxZx_btauhtaue','label_H_ZxZx_btauhtaum','label_H_ZxZx_btauhtauh','label_H_ZxZx_ccee','label_H_ZxZx_ccmm','label_H_ZxZx_cce','label_H_ZxZx_ccm','label_H_ZxZx_cee','label_H_ZxZx_cmm','label_H_ZxZx_cctauhtaue','label_H_ZxZx_cctauhtaum','label_H_ZxZx_cctauhtauh','label_H_ZxZx_ctauhtaue','label_H_ZxZx_ctauhtaum','label_H_ZxZx_ctauhtauh','label_H_ZxZx_ssee','label_H_ZxZx_ssmm','label_H_ZxZx_sse','label_H_ZxZx_ssm','label_H_ZxZx_see','label_H_ZxZx_smm','label_H_ZxZx_sstauhtaue','label_H_ZxZx_sstauhtaum','label_H_ZxZx_sstauhtauh','label_H_ZxZx_stauhtaue','label_H_ZxZx_stauhtaum','label_H_ZxZx_stauhtauh','label_H_ZxZx_qqee','label_H_ZxZx_qqmm','label_H_ZxZx_qqe','label_H_ZxZx_qqm','label_H_ZxZx_qee','label_H_ZxZx_qmm','label_H_ZxZx_qqtauhtaue','label_H_ZxZx_qqtauhtaum','label_H_ZxZx_qqtauhtauh','label_H_ZxZx_qtauhtaue','label_H_ZxZx_qtauhtaum','label_H_ZxZx_qtauhtauh','label_H_ZxZxStar_bbbb','label_H_ZxZxStar_bbcc','label_H_ZxZxStar_bbss','label_H_ZxZxStar_bbqq','label_H_ZxZxStar_cccc','label_H_ZxZxStar_ccss','label_H_ZxZxStar_ccqq','label_H_ZxZxStar_ssss','label_H_ZxZxStar_ssqq','label_H_ZxZxStar_qqqq','label_H_ZxZxStar_bbb','label_H_ZxZxStar_bbc','label_H_ZxZxStar_bbs','label_H_ZxZxStar_bbq','label_H_ZxZxStar_ccb','label_H_ZxZxStar_ccc','label_H_ZxZxStar_ccs','label_H_ZxZxStar_ccq','label_H_ZxZxStar_ssb','label_H_ZxZxStar_ssc','label_H_ZxZxStar_sss','label_H_ZxZxStar_ssq','label_H_ZxZxStar_qqb','label_H_ZxZxStar_qqc','label_H_ZxZxStar_qqs','label_H_ZxZxStar_qqq','label_H_ZxZxStar_bbee','label_H_ZxZxStar_bbmm','label_H_ZxZxStar_bbe','label_H_ZxZxStar_bbm','label_H_ZxZxStar_bee','label_H_ZxZxStar_bmm','label_H_ZxZxStar_bbtauhtaue','label_H_ZxZxStar_bbtauhtaum','label_H_ZxZxStar_bbtauhtauh','label_H_ZxZxStar_btauhtaue','label_H_ZxZxStar_btauhtaum','label_H_ZxZxStar_btauhtauh','label_H_ZxZxStar_ccee','label_H_ZxZxStar_ccmm','label_H_ZxZxStar_cce','label_H_ZxZxStar_ccm','label_H_ZxZxStar_cee','label_H_ZxZxStar_cmm','label_H_ZxZxStar_cctauhtaue','label_H_ZxZxStar_cctauhtaum','label_H_ZxZxStar_cctauhtauh','label_H_ZxZxStar_ctauhtaue','label_H_ZxZxStar_ctauhtaum','label_H_ZxZxStar_ctauhtauh','label_H_ZxZxStar_ssee','label_H_ZxZxStar_ssmm','label_H_ZxZxStar_sse','label_H_ZxZxStar_ssm','label_H_ZxZxStar_see','label_H_ZxZxStar_smm','label_H_ZxZxStar_sstauhtaue','label_H_ZxZxStar_sstauhtaum','label_H_ZxZxStar_sstauhtauh','label_H_ZxZxStar_stauhtaue','label_H_ZxZxStar_stauhtaum','label_H_ZxZxStar_stauhtauh','label_H_ZxZxStar_qqee','label_H_ZxZxStar_qqmm','label_H_ZxZxStar_qqe','label_H_ZxZxStar_qqm','label_H_ZxZxStar_qee','label_H_ZxZxStar_qmm','label_H_ZxZxStar_qqtauhtaue','label_H_ZxZxStar_qqtauhtaum','label_H_ZxZxStar_qqtauhtauh','label_H_ZxZxStar_qtauhtaue','label_H_ZxZxStar_qtauhtaum','label_H_ZxZxStar_qtauhtauh','label_H_HV_aabb','label_H_HV_aacc','label_H_HV_aass','label_H_HV_aaqq','label_H_HV_aabc','label_H_HV_aacs','label_H_HV_aabq','label_H_HV_aacq','label_H_HV_aasq','label_H_HV_aagg','label_H_HV_aaee','label_H_HV_aamm','label_H_HV_aatauhtaue','label_H_HV_aatauhtaum','label_H_HV_aatauhtauh','label_H_HV_aab','label_H_HV_aac','label_H_HV_aas','label_H_HV_aaq','label_H_HV_aag','label_H_HV_aae','label_H_HV_aam','label_H_HV_aataue','label_H_HV_aataum','label_H_HV_aatauh','label_H_HV_abb','label_H_HV_acc','label_H_HV_ass','label_H_HV_aqq','label_H_HV_abc','label_H_HV_acs','label_H_HV_abq','label_H_HV_acq','label_H_HV_asq','label_H_HV_agg','label_H_HV_aee','label_H_HV_amm','label_H_HV_atauhtaue','label_H_HV_atauhtaum','label_H_HV_atauhtauh','label_QCD_bb','label_QCD_cc','label_QCD_b','label_QCD_c','label_QCD_others']"


PREFIX=ak8_MD_inclv10init_manual.ddp4-bs512-lr1e-3.nepoch100.farm221 # 4GPU
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 712 -o num_cls_nodes 355 -o loss_composed_split_reg [[True,True],[True,False]] --num-epochs 100 "

trainopts="--num-workers 3 --fetch-step 1. --data-split-group 320 " # best on farm221
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 --log-file logs/${PREFIX}/val.log "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3beta2 --model-prefix model/${PREFIX}/net_epoch-97_state.pt " # fetch-by-file #### temp try epoch=97

source scripts/train_GloParT_v3.sh run 4,5,6,7 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts
source scripts/train_GloParT_v3.sh run 2 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $valopts
source scripts/train_GloParT_v3.sh dryrun 3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $testopts

// for test
source scripts/train_GloParT_v3.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts --data-train t_ttbar:./datasets/20240909_ak8_UL17_PUPPIv18_v10/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*00.root --data-split-group 2

// eval with ROC
valextopts="-o eval_kw {'roc_kw':{'comp_list':[('Xbb','QCD'),('Xcc','QCD'),('Xcc','Xbb')],'label_inds_map':{'Xbb':[17],'Xcc':[18],'QCD':[350,351,352,353,354]}}} "
source scripts/train_GloParT_v3.sh run 2 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $valopts $valextopts --tensorboard _${PREFIX}.withroc

### updated loss_composed_split_reg logic training (DEPRECATED)

PREFIX=ak8_MD_inclv10init_manual.ddp4-bs512-lr1e-3.nepoch100.fixcompreg.farm221 # 4GPU
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 711 -o num_cls_nodes 355 -o loss_composed_split_reg [[False,True],[True,False]] --num-epochs 100 "

trainopts="--num-workers 3 --fetch-step 1. --data-split-group 320 " # best on farm221
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 --log-file logs/${PREFIX}/val.log"
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

source scripts/train_GloParT_v3.sh run 0,1,2,3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 256 --start-lr 1e-3 $modelopts $trainopts

source scripts/train_GloParT_v3.sh run 4,5,6,7 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts
source scripts/train_GloParT_v3.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT_v3.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts


// for test
source scripts/train_GloParT_v3.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 256 --start-lr 1e-3 $modelopts $trainopts --data-train t_ttbar:./datasets/20240909_ak8_UL17_PUPPIv18_v10/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*00.root --data-split-group 20

### maybe we should not use compose_split_reg at this time.. (DEPRECATED)

// also: add "autorecover" model to the script

PREFIX=ak8_MD_inclv10init_onlysplitreg_manual.ddp4-bs512-lr1e-3.nepoch100.farm221 # 4GPU
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 710 -o num_cls_nodes 355 --num-epochs 100 "

trainopts="--num-workers 3 --fetch-step 1. --data-split-group 320 " # best on farm221
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 --log-file logs/${PREFIX}/val.log"
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

source scripts/train_GloParT_v3.sh autorecover 4,5,6,7 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts

## Testing PUPPIv18 training results

// record the new test file configs

// for run2 UL original
qcd170to300:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/*.root \
qcd300to470:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_300to470_TuneCP5_13TeV_pythia8/*.root \
qcd470to600:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/*.root \
qcd600to800:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_600to800_TuneCP5_13TeV_pythia8/*.root \
qcd800to1000:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8/*.root \
qcd1000to1400:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/*.root \
qcd1400to1800:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/*.root \
qcd1800to2400:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/*.root \
qcd2400to3200:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8/*.root \
qcd3200toinf:./datasets/20230504_ak8_UL17_v8/infer/QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8/*.root \
higlo_part0:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*[0-2].root \
higlo_part1:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*[3-5].root \
higlo_part2:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*[6-9].root \
highi_part0:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*[0-2].root \
highi_part1:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*[3-5].root \
highi_part2:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*[6-9].root \


// for run2 UL re-PUPPI & Run3 2023/2023bpix
run3_2023_qcd170to300:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_170to300_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd300to470:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_300to470_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd470to600:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_470to600_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd600to800:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_600to800_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd800to1000:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_800to1000_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd1000to1400:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_1000to1400_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd1400to1800:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_1400to1800_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd1800to2400:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_1800to2400_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd2400to3200:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_2400to3200_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_qcd3200toinf:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_3200toInf_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023_higlo_part0:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[0-2].root \
run3_2023_higlo_part1:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[3-5].root \
run3_2023_higlo_part2:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[6-9].root \
run3_2023_highi_part0:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[0-2].root \
run3_2023_highi_part1:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[3-5].root \
run3_2023_highi_part2:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[6-9].root \
run3_2023bpix_qcd170to300:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_170to300_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd300to470:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_300to470_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd470to600:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_470to600_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd600to800:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_600to800_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd800to1000:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_800to1000_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd1000to1400:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_1000to1400_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd1400to1800:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_1400to1800_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd1800to2400:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_1800to2400_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd2400to3200:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_2400to3200_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_qcd3200toinf:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_3200toInf_TuneCP5_13p6TeV_pythia8/*.root \
run3_2023bpix_higlo_part0:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[0-2].root \
run3_2023bpix_higlo_part1:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[3-5].root \
run3_2023bpix_higlo_part2:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[6-9].root \
run3_2023bpix_highi_part0:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[0-2].root \
run3_2023bpix_highi_part1:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[3-5].root \
run3_2023bpix_highi_part2:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[6-9].root \
run2_repuppi_qcd170to300:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd300to470:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_300to470_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd470to600:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd600to800:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_600to800_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd800to1000:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd1000to1400:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd1400to1800:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd1800to2400:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd2400to3200:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_qcd3200toinf:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8/*.root \
run2_repuppi_higlo_part0:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[0-2].root \
run2_repuppi_higlo_part1:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[3-5].root \
run2_repuppi_higlo_part2:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_LowPt_narrow/*[6-9].root \
run2_repuppi_highi_part0:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[0-2].root \
run2_repuppi_highi_part1:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[3-5].root \
run2_repuppi_highi_part2:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_HighPt_narrow/*[6-9].root \
run2_repuppi_hwwlo:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_LowPt_narrow/*.root \
run2_repuppi_hwwhi:./datasets/20240824_ak8_Run3_v10/infer_UL17/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_HighPt_narrow/*.root \
run2_repuppi_ttbar:./datasets/20240824_ak8_Run3_v10/infer_UL17/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root \

// old qcd... for recording
run3_2023_qcd_part0:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/*[0-2].root \
run3_2023_qcd_part1:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/*[3-5].root \
run3_2023_qcd_part2:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023/QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/*[6-9].root \
run3_2023bpix_qcd_part0:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/*[0-2].root \
run3_2023bpix_qcd_part1:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/*[3-5].root \
run3_2023bpix_qcd_part2:./datasets/20240824_ak8_Run3_v10/infer_Run3_2023BPix/QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/*[6-9].root \

// reschedule QCD dirs
qcddir=QCD_Pt_170to300_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_300to470_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_470to600_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_600to800_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_800to1000_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 50 99`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done
qcddir=QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8; mkdir ${qcddir}_UL18; for i in `seq 25 61`; do mv ${qcddir}/*-$i.root ${qcddir}_UL18; done

// setup pt-binned QCD samples for Run3
jobid=7187086 # 2023
jobid=7187092 # 2023bpix
qcddir=QCD_Pt_170to300_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 0 7`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_300to470_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 8 15`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_470to600_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 16 23`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_600to800_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 24 31`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_800to1000_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 32 39`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_1000to1400_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 40 47`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_1400to1800_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 48 55`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_1800to2400_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 56 63`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_2400to3200_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 64 71`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;
qcddir=QCD_Pt_3200toInf_TuneCP5_13p6TeV_pythia8; mkdir $qcddir; cd $qcddir; for i in `seq 72 76`; do file="../QCD_Pt_170toInf_ptBinned_TuneCP5_13p6TeV_pythia8/dnnTuples_${jobid}-$i.root"; [[ -f $file ]] && ln -s $file .; done; cd -;

## 24.09.20 beta 3 training

// get sample weights

PREFIX=ak8_MD_inclv10beta3_test.ddp-bs512-lr7e-4.nepoch100
PREFIX=ak8_MD_inclv10beta3.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="--num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

source scripts/train_GloParT_v3.sh run cpu --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 256 --start-lr 7e-4 $modelopts $trainopts --log-file logs/${PREFIX}/pre_train.log


### beta 3 training

label_cls_nodes_v3beta3="['label_H_bb','label_H_cc','label_H_ss','label_H_qq','label_Hp_bc','label_Hm_bc','label_H_bs','label_Hp_cs','label_Hm_cs','label_H_gg','label_H_aa','label_H_ee','label_H_mm','label_H_tauhtaue','label_H_tauhtaum','label_H_tauhtauh','label_Top_bWpcs','label_Top_bWpqq','label_Top_bWpc','label_Top_bWps','label_Top_bWpq','label_Top_bWpev','label_Top_bWpmv','label_Top_bWptauev','label_Top_bWptaumv','label_Top_bWptauhv','label_Top_Wpcs','label_Top_Wpqq','label_Top_Wpev','label_Top_Wpmv','label_Top_Wptauev','label_Top_Wptaumv','label_Top_Wptauhv','label_Top_bWmcs','label_Top_bWmqq','label_Top_bWmc','label_Top_bWms','label_Top_bWmq','label_Top_bWmev','label_Top_bWmmv','label_Top_bWmtauev','label_Top_bWmtaumv','label_Top_bWmtauhv','label_Top_Wmcs','label_Top_Wmqq','label_Top_Wmev','label_Top_Wmmv','label_Top_Wmtauev','label_Top_Wmtaumv','label_Top_Wmtauhv','label_H_WW_cscs','label_H_WW_csqq','label_H_WW_qqqq','label_H_WW_csc','label_H_WW_css','label_H_WW_csq','label_H_WW_qqc','label_H_WW_qqs','label_H_WW_qqq','label_H_WW_csev','label_H_WW_qqev','label_H_WW_csmv','label_H_WW_qqmv','label_H_WW_cstauev','label_H_WW_qqtauev','label_H_WW_cstaumv','label_H_WW_qqtaumv','label_H_WW_cstauhv','label_H_WW_qqtauhv','label_H_WxWx_cscs','label_H_WxWx_csqq','label_H_WxWx_qqqq','label_H_WxWx_csc','label_H_WxWx_css','label_H_WxWx_csq','label_H_WxWx_qqc','label_H_WxWx_qqs','label_H_WxWx_qqq','label_H_WxWx_csev','label_H_WxWx_qqev','label_H_WxWx_csmv','label_H_WxWx_qqmv','label_H_WxWx_cstauev','label_H_WxWx_qqtauev','label_H_WxWx_cstaumv','label_H_WxWx_qqtaumv','label_H_WxWx_cstauhv','label_H_WxWx_qqtauhv','label_H_WxWxStar_cscs','label_H_WxWxStar_csqq','label_H_WxWxStar_qqqq','label_H_WxWxStar_csc','label_H_WxWxStar_css','label_H_WxWxStar_csq','label_H_WxWxStar_qqc','label_H_WxWxStar_qqs','label_H_WxWxStar_qqq','label_H_WxWxStar_csev','label_H_WxWxStar_qqev','label_H_WxWxStar_csmv','label_H_WxWxStar_qqmv','label_H_WxWxStar_cstauev','label_H_WxWxStar_qqtauev','label_H_WxWxStar_cstaumv','label_H_WxWxStar_qqtaumv','label_H_WxWxStar_cstauhv','label_H_WxWxStar_qqtauhv','label_H_ZZ_bbbb','label_H_ZZ_bbcc','label_H_ZZ_bbss','label_H_ZZ_bbqq','label_H_ZZ_cccc','label_H_ZZ_ccss','label_H_ZZ_ccqq','label_H_ZZ_ssss','label_H_ZZ_ssqq','label_H_ZZ_qqqq','label_H_ZZ_bbb','label_H_ZZ_bbc','label_H_ZZ_bbs','label_H_ZZ_bbq','label_H_ZZ_ccb','label_H_ZZ_ccc','label_H_ZZ_ccs','label_H_ZZ_ccq','label_H_ZZ_ssb','label_H_ZZ_ssc','label_H_ZZ_sss','label_H_ZZ_ssq','label_H_ZZ_qqb','label_H_ZZ_qqc','label_H_ZZ_qqs','label_H_ZZ_qqq','label_H_ZZ_bbee','label_H_ZZ_bbmm','label_H_ZZ_bbe','label_H_ZZ_bbm','label_H_ZZ_bee','label_H_ZZ_bmm','label_H_ZZ_bbtauhtaue','label_H_ZZ_bbtauhtaum','label_H_ZZ_bbtauhtauh','label_H_ZZ_btauhtaue','label_H_ZZ_btauhtaum','label_H_ZZ_btauhtauh','label_H_ZZ_ccee','label_H_ZZ_ccmm','label_H_ZZ_cce','label_H_ZZ_ccm','label_H_ZZ_cee','label_H_ZZ_cmm','label_H_ZZ_cctauhtaue','label_H_ZZ_cctauhtaum','label_H_ZZ_cctauhtauh','label_H_ZZ_ctauhtaue','label_H_ZZ_ctauhtaum','label_H_ZZ_ctauhtauh','label_H_ZZ_ssee','label_H_ZZ_ssmm','label_H_ZZ_sse','label_H_ZZ_ssm','label_H_ZZ_see','label_H_ZZ_smm','label_H_ZZ_sstauhtaue','label_H_ZZ_sstauhtaum','label_H_ZZ_sstauhtauh','label_H_ZZ_stauhtaue','label_H_ZZ_stauhtaum','label_H_ZZ_stauhtauh','label_H_ZZ_qqee','label_H_ZZ_qqmm','label_H_ZZ_qqe','label_H_ZZ_qqm','label_H_ZZ_qee','label_H_ZZ_qmm','label_H_ZZ_qqtauhtaue','label_H_ZZ_qqtauhtaum','label_H_ZZ_qqtauhtauh','label_H_ZZ_qtauhtaue','label_H_ZZ_qtauhtaum','label_H_ZZ_qtauhtauh','label_H_ZxZx_bbbb','label_H_ZxZx_bbcc','label_H_ZxZx_bbss','label_H_ZxZx_bbqq','label_H_ZxZx_cccc','label_H_ZxZx_ccss','label_H_ZxZx_ccqq','label_H_ZxZx_ssss','label_H_ZxZx_ssqq','label_H_ZxZx_qqqq','label_H_ZxZx_bbb','label_H_ZxZx_bbc','label_H_ZxZx_bbs','label_H_ZxZx_bbq','label_H_ZxZx_ccb','label_H_ZxZx_ccc','label_H_ZxZx_ccs','label_H_ZxZx_ccq','label_H_ZxZx_ssb','label_H_ZxZx_ssc','label_H_ZxZx_sss','label_H_ZxZx_ssq','label_H_ZxZx_qqb','label_H_ZxZx_qqc','label_H_ZxZx_qqs','label_H_ZxZx_qqq','label_H_ZxZx_bbee','label_H_ZxZx_bbmm','label_H_ZxZx_bbe','label_H_ZxZx_bbm','label_H_ZxZx_bee','label_H_ZxZx_bmm','label_H_ZxZx_bbtauhtaue','label_H_ZxZx_bbtauhtaum','label_H_ZxZx_bbtauhtauh','label_H_ZxZx_btauhtaue','label_H_ZxZx_btauhtaum','label_H_ZxZx_btauhtauh','label_H_ZxZx_ccee','label_H_ZxZx_ccmm','label_H_ZxZx_cce','label_H_ZxZx_ccm','label_H_ZxZx_cee','label_H_ZxZx_cmm','label_H_ZxZx_cctauhtaue','label_H_ZxZx_cctauhtaum','label_H_ZxZx_cctauhtauh','label_H_ZxZx_ctauhtaue','label_H_ZxZx_ctauhtaum','label_H_ZxZx_ctauhtauh','label_H_ZxZx_ssee','label_H_ZxZx_ssmm','label_H_ZxZx_sse','label_H_ZxZx_ssm','label_H_ZxZx_see','label_H_ZxZx_smm','label_H_ZxZx_sstauhtaue','label_H_ZxZx_sstauhtaum','label_H_ZxZx_sstauhtauh','label_H_ZxZx_stauhtaue','label_H_ZxZx_stauhtaum','label_H_ZxZx_stauhtauh','label_H_ZxZx_qqee','label_H_ZxZx_qqmm','label_H_ZxZx_qqe','label_H_ZxZx_qqm','label_H_ZxZx_qee','label_H_ZxZx_qmm','label_H_ZxZx_qqtauhtaue','label_H_ZxZx_qqtauhtaum','label_H_ZxZx_qqtauhtauh','label_H_ZxZx_qtauhtaue','label_H_ZxZx_qtauhtaum','label_H_ZxZx_qtauhtauh','label_H_ZxZxStar_bbbb','label_H_ZxZxStar_bbcc','label_H_ZxZxStar_bbss','label_H_ZxZxStar_bbqq','label_H_ZxZxStar_cccc','label_H_ZxZxStar_ccss','label_H_ZxZxStar_ccqq','label_H_ZxZxStar_ssss','label_H_ZxZxStar_ssqq','label_H_ZxZxStar_qqqq','label_H_ZxZxStar_bbb','label_H_ZxZxStar_bbc','label_H_ZxZxStar_bbs','label_H_ZxZxStar_bbq','label_H_ZxZxStar_ccb','label_H_ZxZxStar_ccc','label_H_ZxZxStar_ccs','label_H_ZxZxStar_ccq','label_H_ZxZxStar_ssb','label_H_ZxZxStar_ssc','label_H_ZxZxStar_sss','label_H_ZxZxStar_ssq','label_H_ZxZxStar_qqb','label_H_ZxZxStar_qqc','label_H_ZxZxStar_qqs','label_H_ZxZxStar_qqq','label_H_ZxZxStar_bbee','label_H_ZxZxStar_bbmm','label_H_ZxZxStar_bbe','label_H_ZxZxStar_bbm','label_H_ZxZxStar_bee','label_H_ZxZxStar_bmm','label_H_ZxZxStar_bbtauhtaue','label_H_ZxZxStar_bbtauhtaum','label_H_ZxZxStar_bbtauhtauh','label_H_ZxZxStar_btauhtaue','label_H_ZxZxStar_btauhtaum','label_H_ZxZxStar_btauhtauh','label_H_ZxZxStar_ccee','label_H_ZxZxStar_ccmm','label_H_ZxZxStar_cce','label_H_ZxZxStar_ccm','label_H_ZxZxStar_cee','label_H_ZxZxStar_cmm','label_H_ZxZxStar_cctauhtaue','label_H_ZxZxStar_cctauhtaum','label_H_ZxZxStar_cctauhtauh','label_H_ZxZxStar_ctauhtaue','label_H_ZxZxStar_ctauhtaum','label_H_ZxZxStar_ctauhtauh','label_H_ZxZxStar_ssee','label_H_ZxZxStar_ssmm','label_H_ZxZxStar_sse','label_H_ZxZxStar_ssm','label_H_ZxZxStar_see','label_H_ZxZxStar_smm','label_H_ZxZxStar_sstauhtaue','label_H_ZxZxStar_sstauhtaum','label_H_ZxZxStar_sstauhtauh','label_H_ZxZxStar_stauhtaue','label_H_ZxZxStar_stauhtaum','label_H_ZxZxStar_stauhtauh','label_H_ZxZxStar_qqee','label_H_ZxZxStar_qqmm','label_H_ZxZxStar_qqe','label_H_ZxZxStar_qqm','label_H_ZxZxStar_qee','label_H_ZxZxStar_qmm','label_H_ZxZxStar_qqtauhtaue','label_H_ZxZxStar_qqtauhtaum','label_H_ZxZxStar_qqtauhtauh','label_H_ZxZxStar_qtauhtaue','label_H_ZxZxStar_qtauhtaum','label_H_ZxZxStar_qtauhtauh','label_H_HV_aabb','label_H_HV_aacc','label_H_HV_aass','label_H_HV_aaqq','label_H_HV_aabc','label_H_HV_aacs','label_H_HV_aabq','label_H_HV_aacq','label_H_HV_aasq','label_H_HV_aagg','label_H_HV_aaee','label_H_HV_aamm','label_H_HV_aatauhtaue','label_H_HV_aatauhtaum','label_H_HV_aatauhtauh','label_H_HV_aab','label_H_HV_aac','label_H_HV_aas','label_H_HV_aaq','label_H_HV_aag','label_H_HV_aae','label_H_HV_aam','label_H_HV_aataue','label_H_HV_aataum','label_H_HV_aatauh','label_H_HV_abb','label_H_HV_acc','label_H_HV_ass','label_H_HV_aqq','label_H_HV_abc','label_H_HV_acs','label_H_HV_abq','label_H_HV_acq','label_H_HV_asq','label_H_HV_agg','label_H_HV_aee','label_H_HV_amm','label_H_HV_atauhtaue','label_H_HV_atauhtaum','label_H_HV_atauhtauh','label_QCD_bb','label_QCD_cc','label_QCD_b','label_QCD_c','label_QCD_others']"

PREFIX=ak8_MD_inclv10beta3_manual.ddp4-bs512-lr1e-3.nepoch100.farm221 # 4GPU
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 750 -o num_cls_nodes 374 -o loss_composed_split_reg [[True,True],[True,False]] --num-epochs 100 "

trainopts="--num-workers 3 --fetch-step 1. --data-split-group 320 " # best on farm221
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 --log-file logs/${PREFIX}/val.log "
valextopts="-o eval_kw {'roc_kw':{'comp_list':[('Xbb','QCD'),('Xcc','QCD'),('Xcc','Xbb')],'label_inds_map':{'Xbb':[0],'Xcc':[1],'QCD':[369,370,371,372,373]}}} "
testopts="--run-mode test --num-workers 2 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3beta3 --model-prefix model/${PREFIX}/net_epoch-97_state.pt " # fetch-by-file ##!! use model 97

source scripts/train_GloParT_v3.sh run 0,1,2,3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts
source scripts/train_GloParT_v3.sh run 2 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $valopts $valextopts
source scripts/train_GloParT_v3.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $testopts

// for test
source scripts/train_GloParT_v3.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts --data-train t_ttbar:./datasets/20240909_ak8_UL17_PUPPIv18_v10/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*00.root --data-split-group 2

source scripts/train_GloParT_v3.sh run 3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $valopts --load-epoch 59 --samples-per-epoch-val 100000000 --log-file logs/${PREFIX}/val.checknan.log --tensorboard _${PREFIX}.checknan

// test step with epoch=67
specialtestopts="--run-mode test --num-workers 2 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3beta3 --model-prefix model/${PREFIX}/net_epoch-67_state.pt --predict-output predict/$PREFIX.epoch67/pred.root" # fetch-by-file ##!! use model 67
source scripts/train_GloParT_v3.sh dryrun 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $specialtestopts


### beta 3 training (original QCD) -> card + bash script changes
// card: QCD weight back to 1..

// get sample weights

PREFIX=ak8_MD_inclv10beta3_test.ddp-bs512-lr7e-4.nepoch100
PREFIX=ak8_MD_inclv10beta3.ddp-bs512-lr7e-4.nepoch100
PREFIX=ak8_MD_inclv10beta3_origQCD.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="--num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3pre " # fetch-by-file

source scripts/train_GloParT_v3_origQCD.sh run cpu --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 256 --start-lr 7e-4 $modelopts $trainopts --log-file logs/${PREFIX}/pre_train.log --print

// training

PREFIX=ak8_MD_inclv10beta3_origQCD_manual.ddp4-bs512-lr1e-3.nepoch100.ihep # 4GPU
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 750 -o num_cls_nodes 374 -o loss_composed_split_reg [[True,True],[True,False]] --num-epochs 100 "

trainopts="--num-workers 4 --fetch-step 1. --data-split-group 320 " # on ihep
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 --log-file logs/${PREFIX}/val.log "
valextopts="-o eval_kw {'roc_kw':{'comp_list':[('Xbb','QCD'),('Xcc','QCD'),('Xcc','Xbb')],'label_inds_map':{'Xbb':[0],'Xcc':[1],'QCD':[369,370,371,372,373]}}} "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3beta3 " # fetch-by-file

source scripts/train_GloParT_v3_origQCD.sh run 0,1,2,3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts
source scripts/train_GloParT_v3_origQCD.sh run 3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $valopts $valextopts
source scripts/train_GloParT_v3_origQCD.sh dryrun 3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $testopts

// test step with epoch=67
specialtestopts="--run-mode test --num-workers 2 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3beta3 --model-prefix model/${PREFIX}/net_epoch-67_state.pt --predict-output predict/$PREFIX.epoch67/pred.root" # fetch-by-file ##!! use model 67
source scripts/train_GloParT_v3_origQCD.sh dryrun 1 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $specialtestopts

// continue training from epoch=69 towards 120

PREFIX=ak8_MD_inclv10beta3_origQCD_manual.ddp4-bs512-lr1e-3.nepoch120.ihep # 4GPU -> change to 120
config=./data_new/inclv10/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 750 -o num_cls_nodes 374 -o loss_composed_split_reg [[True,True],[True,False]] --num-epochs 120 "

trainopts="--num-workers 3 --fetch-step 1. --data-split-group 320 " # on farm221
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 --log-file logs/${PREFIX}/val.log "
valextopts="-o eval_kw {'roc_kw':{'comp_list':[('Xbb','QCD'),('Xcc','QCD'),('Xcc','Xbb')],'label_inds_map':{'Xbb':[0],'Xcc':[1],'QCD':[369,370,371,372,373]}}} "
testopts="--run-mode test --num-workers 3 --data-split-group 1 -o label_cls_nodes $label_cls_nodes_v3beta3 " # fetch-by-file

source scripts/train_GloParT_v3_origQCD.sh run 0,1,2,3 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 1e-3 $modelopts $trainopts --load-epoch 69