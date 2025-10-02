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

NDIV=50 # 10 or 50
trainset_gghh=$(ls -v ${DATADIR}/sm_4j/HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple/*.root | head -n $(($(ls -1 ${DATADIR}/sm_4j/HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple | wc -l) / $NDIV)) | sed 's/^/gghh:/' | paste -sd' ' -)
trainset_qcd=$(ls -v ${DATADIR}/sm_4j/QCD_DelphesHH4JTrig_merged_ntuple/*.root | head -n $(($(ls -1 ${DATADIR}/sm_4j/QCD_DelphesHH4JTrig_merged_ntuple | wc -l) / $NDIV)) | sed 's/^/qcd:/' | paste -sd' ' -)
trainset_ttbar=$(ls -v ${DATADIR}/sm_incl_derived_4j3bor2b/TTbar_ntuple/*.root | head -n $(($(ls -1 ${DATADIR}/sm_incl_derived_4j3bor2b/TTbar_ntuple | wc -l) / $NDIV)) | sed 's/^/ttbar:/' | paste -sd' ' -)

if [ $NDIV -eq 50 ]; then
    samples_per_epoch=$((1000 * 1024 / $NGPUS))
    samples_per_epoch_val=$((250 * 1024 / $NGPUS))
    num_epochs=40
elif [ $NDIV -eq 10 ]; then
    samples_per_epoch=$((4000 * 1024 / $NGPUS))
    samples_per_epoch_val=$((1000 * 1024 / $NGPUS))
    num_epochs=40
elif [ $NDIV -eq 1 ]; then
    samples_per_epoch=$((10000 * 1024 / $NGPUS))
    samples_per_epoch_val=$((2500 * 1024 / $NGPUS))
    num_epochs=80
else
    echo "NDIV must be 50, 10 or 1"
    exit 1
fi

ARG="--network-config networks/pheno2/example_ParticleTransformer.py -o num_classes 2 -o embed_dims [256,1024,256] -o pair_embed_dims [64,64,64] -o num_heads 16 -o fc_params [(1024,0.1)] \
--use-amp --batch-size 256 --start-lr 2.5e-4 --samples-per-epoch $samples_per_epoch --samples-per-epoch-val $samples_per_epoch_val --num-epochs $num_epochs --optimizer ranger \
--num-workers 5 --fetch-step 1.0 --data-split-num 500 \
--data-train $trainset_gghh $trainset_qcd $trainset_ttbar
--data-config $config \
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
    unset CUDA_VISIBLE_DEVICES
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
