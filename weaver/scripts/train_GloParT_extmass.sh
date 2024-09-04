#!/bin/bash -x

RUN=$1
GPUS=$2

if [ -z $GPUS ]; then
    echo "Usage: $0 <ngpu>"
    exit 1
fi
NGPUS=$(echo $GPUS | tr "," "\n" | wc -l)

cmdlineopts="${@:3}"

current_dir=`pwd`
if [[ "$current_dir" != *"weaver-core/weaver" ]]; then
    echo "Please run this script from the weaver directory"
    exit 1
fi

# v2 default command
## remember: remove all single-quote characters
ARG="--run-mode train-only \
--train-mode hybrid -o three_coll True -o loss_gamma 5 -o fc_params [(1024,0.1)] -o embed_dims [256,1024,256] -o pair_embed_dims [128,128,128] -o num_heads 16 \
--use-amp --batch-size 256 --start-lr 2e-3 --num-epochs 50 --optimizer ranger --fetch-step 0.01 --data-split-group 3 \
--data-train \
t_qcd170to300:./datasets/20230504_ak8_UL17_v8/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/*.root \
t_qcd300to470:./datasets/20230504_ak8_UL17_v8/QCD_Pt_300to470_TuneCP5_13TeV_pythia8/*.root \
t_qcd470to600:./datasets/20230504_ak8_UL17_v8/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/*.root \
t_qcd600to800:./datasets/20230504_ak8_UL17_v8/QCD_Pt_600to800_TuneCP5_13TeV_pythia8/*.root \
t_qcd800to1000:./datasets/20230504_ak8_UL17_v8/QCD_Pt_800to1000_TuneCP5_13TeV_pythia8/*.root \
t_qcd1000to1400:./datasets/20230504_ak8_UL17_v8/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/*.root \
t_qcd1400to1800:./datasets/20230504_ak8_UL17_v8/QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/*.root \
t_qcd1800to2400:./datasets/20230504_ak8_UL17_v8/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/*.root \
t_qcd2400to3200:./datasets/20230504_ak8_UL17_v8/QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8/*.root \
t_qcd3200toinf:./datasets/20230504_ak8_UL17_v8/QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8/*.root \
t_ttbar:./datasets/20230504_ak8_UL17_v8/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root \
t_h2p:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root \
t_hpm2p:./datasets/20230504_ak8_UL17_v8/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root \
t_hww:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root \
t_hwxwx:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root \
t_hzz:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root \
t_hzxzx:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root \
t_ttbarhm:./datasets/20230504_ak8_UL17_v8_ext1/Spin0ToTT_VariableMass_WhadOrlep_MX-Var_MH-260to650/*.root \
t_h2phm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4QGluLTau_MX-Var_MH-260to650/*.root \
t_hpm2phm:./datasets/20230504_ak8_UL17_v8_ext1/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-Var_MH-260to650/*.root \
t_hwwhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4W_MX-Var_MH-260to650_JHUVariableWMass/*.root \
t_hwxwxhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4W_MX-Var_MH-260to650_JHUVariableWMass2DMesh/*.root \
t_hzzhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4Z_MX-Var_MH-260to650_JHUVariableZMass/*.root \
t_hzxzxhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4Z_MX-Var_MH-260to650_JHUVariableZMass2DMesh/*.root \
--data-test \
hww:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_M-1000_narrow/*.root \
higgs2p:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_M-1000_narrow/*.root \
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
ttbar:./datasets/20230504_ak8_UL17_v8/infer/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root \
higlo_part0:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*[0-2].root \
higlo_part1:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*[3-5].root \
higlo_part2:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-8_narrow/*[6-9].root \
highi_part0:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*[0-2].root \
highi_part1:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*[3-5].root \
highi_part2:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4QGluLTau_MH-50-125-250-300_RatioMGMH-20_narrow/*[6-9].root \
hwwlo:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-8_narrow/*.root \
hwwhi:./datasets/20230504_ak8_UL17_v8/infer/GluGluToBulkGravitonToHHTo4W_JHUGen_MH-50-125-250-300_RatioMGMH-20_narrow/*.root \
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} --num-workers 5 \
--network-config networks/example_ParticleTransformer2023Tagger_hybrid.py \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "

if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

echo "Run argument: $ARG"

# if GPUS is an integer
if [ $GPUS -eq $GPUS 2>/dev/null ]; then
    python train.py --gpus $GPUS $ARG $cmdlineopts
else
    # GPU list is separated by comma
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS train.py --backend nccl $ARG $cmdlineopts
fi
