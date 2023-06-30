# call ./3d_shapesDiscriminativePosthoc.sh <corr> -1 <runid> <attribution in {grad, ig, sg}>
ATTRIBUTION=$4
ITERS=10000
LOGPATH="$LOGDIR/DiscriminativeR$3_Given$2_Corr-$1"
CHECKPOINT="$CHECKPOINTDIR/DiscriminativeR$3_Given$2_Corr-$1/best"
EVAL_BATCHES=10000
echo $CHECKPOINT
# Optimizer criteria
export PYTHONPATH="."; python3 discover_concepts.py --checkpoint $CHECKPOINT --attribution $ATTRIBUTION --batchsize=48 --max_iter $ITERS --learning_rate 0.001 --optimizer rmsprop \
--disjoint true --norm true --logdir $LOGPATH --dsetdir $DSET_PATH --num_directions 4 --alg Discriminative --sigma_corr $1 --z_dim 6 \
--evaluation_metric dirscore_bin sap_score dci factor_vae_metric --evaliter 1000 --dset shapes3d_noisy --eval_batches $EVAL_BATCHES
export PYTHONPATH="."; python3 discover_concepts.py --checkpoint $CHECKPOINT --attribution $ATTRIBUTION --batchsize=48 --max_iter $ITERS --learning_rate 0.001 --optimizer rmsprop \
--disjoint false --norm true --logdir $LOGPATH --dsetdir $DSET_PATH --num_directions 4 --alg Discriminative --sigma_corr $1 --z_dim 6 \
--evaluation_metric dirscore_bin sap_score dci factor_vae_metric --evaliter 1000 --dset shapes3d_noisy --eval_batches $EVAL_BATCHES
