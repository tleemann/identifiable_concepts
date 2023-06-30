# Call this script with parameters: ./3dshapes_Posthoc.sh <model_name> <attr> <corr> <fact1> <fact2>
ATTRIBUTION=$2
MODELNAME=$1
CORR=$3
ITERS=20000
EVALBATCHES=1000
CHECKPOINT="${CHECKPOINTDIR}/${MODELNAME}/last"
echo $CHECKPOINT
echo "Correlating components $4 and $5"
# Optimizer criteria
python3 discover_concepts.py --checkpoint $CHECKPOINT --attribution $ATTRIBUTION --batchsize=48 --max_iter $ITERS --learning_rate 0.0001 --optimizer rmsprop \
--eval_batches $EVALBATCHES --evaliter 4000 --evaluation_metric dci factor_vae_metric sap_score --disjoint true --z_dim 6 \
--norm true --logdir $LOGDIR/${MODELNAME} --dsetdir $DSET_PATH --alg BetaVAE --sigma_corr $CORR --corr_feature1 $4 --corr_feature2 $5
python3 discover_concepts.py --checkpoint $CHECKPOINT --attribution $ATTRIBUTION --batchsize=48 --max_iter $ITERS --learning_rate 0.0001 --optimizer rmsprop \
--eval_batches $EVALBATCHES --evaliter 4000 --evaluation_metric dci factor_vae_metric sap_score --disjoint false --z_dim 6 \
--norm true --logdir $LOGDIR/${MODELNAME} --dsetdir $DSET_PATH --alg BetaVAE --sigma_corr $CORR --corr_feature1 $4 --corr_feature2 $5
