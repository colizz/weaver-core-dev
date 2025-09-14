#!/bin/bash -x

RUN=$1
GPUS=$2

if [ -z $GPUS ]; then
    echo "Usage: $0 <ngpu>"
    exit 1
fi
NGPUS=$(echo $GPUS | tr "," "\n" | wc -l)
echo $NGPUS
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
t_gammabkg:/home/olympus/pancy/fine-tune/weaver-core/weaver/datasets/GJet_PT-170toInf_DoubleEMEnriched_MGG-80_TuneCP5_13TeV_pythia8/*.root \
t_haahm:/home/olympus/pancy/fine-tune/weaver-core/weaver/datasets/20240929_ak8_UL17_v10/BulkGravitonToHHTo4A_MX-Var_MH-260to650/*.root \
t_haa:/home/olympus/pancy/fine-tune/weaver-core/weaver/datasets/20240929_ak8_UL17_v10/BulkGravitonToHHTo4A_MX-600to6000_MH-15to250/*.root \
t_qcd:/home/olympus/pancy/fine-tune/weaver-core/weaver/datasets/QCD_Pt_170toInf_ptBinned_TuneCP5_13TeV_pythia8_remix_50files/*.root \
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
