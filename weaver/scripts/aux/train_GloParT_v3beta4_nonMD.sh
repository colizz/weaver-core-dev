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
## main params to be overriden by cmdlineopts: --run-mode, --train-mode, -o finetune_kw {..} etc
ARG="--run-mode train --train-mode hybrid \
-o num_nodes 750 -o num_cls_nodes 374 -o use_swiglu_config True -o use_pair_norm_config True \
-o fc_params [(2048,0.1)] -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o num_layers 12 \
-o reg_kw {'gamma':5.,'composed_split_reg':[True,False],'use_resid':True} \
--use-amp --batch-size 512 --start-lr 7e-4 --num-epochs 100 --optimizer ranger \
--num-workers 8 --fetch-step 1. --data-split-num 250 \
--network-config networks/example_ParticleTransformer2024PlusTagger_unified2.py \
--data-train \
t_qcd:./datasets/20230504_ak8_UL17_v8/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8/*.root \
t_smttbar:./datasets/20230826_ak8_UL17_v9/v9-nonMD/ZprimeToTT_M1200to4500_W12to45_TuneCP2_PSweights/*.root \
t_smw:./datasets/20230826_ak8_UL17_v9/v9-nonMD/BulkGravToWWToWhadWhad_narrow_M-500to4500_TuneCP5_13TeV-madgraph-pythia/*.root \
t_smz:./datasets/20230826_ak8_UL17_v9/v9-nonMD/BulkGravToZZToZhadZhad_narrow_M-500to4500_TuneCP5_13TeV-madgraph-pythia/*.root \
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
zhi:./datasets/20230504_ak8_UL17_v8/infer/BulkGravToZZToZhadZhad_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia/*.root \
zlo:./datasets/20230504_ak8_UL17_v8/infer/BulkGravToZZToZhadZhad_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia/*.root \
whi:./datasets/20230504_ak8_UL17_v8/infer/BulkGravToWWToWhadWhad_narrow_M-2500_TuneCP5_13TeV-madgraph-pythia/*.root \
wlo:./datasets/20230504_ak8_UL17_v8/infer/BulkGravToWWToWhadWhad_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia/*.root \
--samples-per-epoch $((500 * 512 / $NGPUS)) --samples-per-epoch-val $((500 * 512)) \
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
        # match model/${PREFIX}/net/net_epoch-(\d+)_state.pt and extract the maximum epoch number
        maxepoch=$(ls model/${PREFIX}/net_epoch-*.pt | sed -n s/.*net_epoch-\([0-9]*\)_state.pt/\1/p | sort -n | tail -n 1)
        if [ -z $maxepoch ]; then
            epochopts=""
        else
            epochopts="--load-epoch $maxepoch"
            echo "Resuming from epoch $maxepoch"
        fi
        sleep 10
    done
fi
