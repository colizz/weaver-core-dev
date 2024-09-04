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

PREFIX=ak8_MD_inclv8std_rmhbs_manual.useamp.large_fc2048.pemb64_block10.gm5.ddp-bs512-lr7e-4.nepoch100
config=./data_new/inclv7plus_extmass/${PREFIX%%.*}.yaml

modelopts="-o num_nodes 626 -o num_cls_nodes 313 -o loss_split_reg True -o fc_params [(2048,0.1)] -o use_swiglu_config True -o use_pair_norm_config True -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 10 --num-epochs 100 "

trainopts="--num-workers 8 --fetch-step 1. --data-split-group 250 " # large memory setup
valopts="--run-mode val --num-workers 20 --fetch-step 1. --data-split-group 125 "
testopts="--run-mode test --num-workers 3 --data-split-group 1 " # fetch-by-file

source scripts/train_GloParT_extmass.sh run 5,6,7 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $trainopts
source scripts/train_GloParT_extmass.sh run 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $valopts
source scripts/train_GloParT_extmass.sh dryrun 0 --network-config networks/example_ParticleTransformer2024PlusTagger_unified.py --batch-size 512 --start-lr 7e-4 $modelopts $testopts


========================================================================================================
## Mid-term stage-3: using 100-epoch version to do non-MD transfer learning

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
