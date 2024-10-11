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

# changed to v3beta4 default command
## remember: remove all single-quote characters
ARG="--run-mode train --train-mode hybrid \
-o num_nodes 750 -o num_cls_nodes 374 -o use_swiglu_config True -o use_pair_norm_config True \
-o fc_params [(2048,0.1)] -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 12 \
-o reg_kw {'gamma':5.,'composed_split_reg':[True,False],'as_resid_of':[0]} \
--use-amp --batch-size 512 --start-lr 7e-4 --num-epochs 100 --optimizer ranger \
--num-workers 8 --fetch-step 1. --data-split-num 250 \
--network-config networks/example_ParticleTransformer2024PlusTagger_unified2.py \
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
t_ttbar:./datasets/20240929_ak8_UL17_v10/Spin0ToTT_VariableMass_WhadOrlep_MX-600to6000_MH-15to250/*.root \
t_h2p:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4QGluLTau_MX-600to6000_MH-15to250/*.root \
t_hpm2p:./datasets/20240929_ak8_UL17_v10/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-600to6000_MH-15to250/*.root \
t_hww:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass/*.root \
t_hwxwx:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4W_MX-600to6000_MH-15to250_JHUVariableWMass2DMesh/*.root \
t_hzz:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass/*.root \
t_hzxzx:./datasets/20230504_ak8_UL17_v8/BulkGravitonToHHTo4Z_MX-600to6000_MH-15to250_JHUVariableZMass2DMesh/*.root \
t_ttbarhm:./datasets/20240929_ak8_UL17_v10/Spin0ToTT_VariableMass_WhadOrlep_MX-Var_MH-260to650/*.root \
t_h2phm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4QGluLTau_MX-Var_MH-260to650/*.root \
t_hpm2phm:./datasets/20240929_ak8_UL17_v10/DiH1OrHpm_2HDM_HpmToBC_HpmToCS_H1ToBS_HT-Var_MH-260to650/*.root \
t_hwwhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4W_MX-Var_MH-260to650_JHUVariableWMass/*.root \
t_hwxwxhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4W_MX-Var_MH-260to650_JHUVariableWMass2DMesh/*.root \
t_hzzhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4Z_MX-Var_MH-260to650_JHUVariableZMass/*.root \
t_hzxzxhm:./datasets/20230504_ak8_UL17_v8_ext1/BulkGravitonToHHTo4Z_MX-Var_MH-260to650_JHUVariableZMass2DMesh/*.root \
t_haa:./datasets/20240929_ak8_UL17_v10/BulkGravitonToHHTo4A_MX-600to6000_MH-15to250/*.root \
t_haa4p:./datasets/20240929_ak8_UL17_v10/H3ToHHToWHorZH_HToAA_MX-600to6000_MH-15to250/*.root \
t_haahm:./datasets/20240929_ak8_UL17_v10/BulkGravitonToHHTo4A_MX-Var_MH-260to650/*.root \
t_haa4phm:./datasets/20240929_ak8_UL17_v10/H3ToHHToWHorZH_HToAA_MX-Var_MH-260to650/*.root \
--data-test \
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
run2_repuppi_qcd470to600:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_470to600_TuneCP5_13TeV_pythia8_UL18/*.root \
run2_repuppi_qcd600to800:./datasets/20240824_ak8_Run3_v10/infer_UL17/QCD_Pt_600to800_TuneCP5_13TeV_pythia8_UL18/*.root \
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
--samples-per-epoch $((15000 * 512 / $NGPUS)) --samples-per-epoch-val $((1000 * 512)) \
--data-config ${config} \
--model-prefix model/${PREFIX}/net \
--predict-output predict/$PREFIX/pred.root "

if [ $RUN == "dryrun" ]; then
    echo "Dryrun mode"
elif [ $RUN == "run" ] || [ $RUN == "autorecover" ]; then
    ARG="$ARG --log-file logs/${PREFIX}/train.log --tensorboard _${PREFIX} "
else
    exit 1
fi

if [ $GPUS == "cpu" ]; then
    cmd="python train.py $ARG $cmdlineopts "
elif [ $GPUS -eq $GPUS 2>/dev/null ]; then
    # if GPUS is an integer
    cmd="python train.py --gpus $GPUS $ARG $cmdlineopts "
else
    # GPU list is separated by comma
    export CUDA_VISIBLE_DEVICES=$GPUS
    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS train.py --backend nccl $ARG $cmdlineopts "
fi

echo Run command: $cmd

if [ $RUN == "dryrun" ] || [ $RUN == "run" ]; then
    $cmd
elif [ $RUN == "autorecover" ]; then
    epochopts=""
    # if the training is halted, resume from the last epoch
    while true; do
        $cmd $epochopts
        ret=$?
        if [ $ret -eq 0 ]; then
            break
        fi
        echo "Error: return code $ret"
        # match model/${PREFIX}/net_epoch-(\d+)_state.pt and extract the maximum epoch number
        maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n 's/.*net_epoch-\([0-9]*\)_state.pt/\1/p' | sort -n | tail -n 1)
        if [ -z $maxepoch ]; then
            epochopts=""
        else
            epochopts="--load-epoch $maxepoch"
            echo "Resuming from epoch $maxepoch"
        fi
        sleep 10
    done
fi
