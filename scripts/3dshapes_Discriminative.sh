#! /bin/sh
# Call using 3dshapes_Discriminative.sh <corr> <given factors or -1 in none given> <runid>
echo "Using corr=$1"
if [ -z ${3+x} ]; then RUNID="0"; else RUNID=$3; fi
NAME="DiscriminativeR${RUNID}_Given$2_Corr-$1"

FACTORS="-1"
if [ $2 -gt -1 ]; 
then
FACTORS=""
for i in $(seq 0 $2);
do
   FACTORS="${FACTORS} ${i}"
done
fi
echo "Annotating factors $FACTORS"
export PYTHONPATH="."; python3 main.py \
--name=$NAME \
--alg=Discriminative \
--dset_dir=$DSET_PATH \
--dset_name=shapes3d_noisy \
--classifier=GAPClassifier \
--sigma_corr $1 \
--corr_feature1 1 \
--corr_feature2 0 \
--z_dim=6 --w_kld=1 --use_wandb=false \
--max_epoch 5 \
--max_iter 10000 \
--oversampling_factor 5 \
--labeling_fn shapes3d_binary_labelfunction_eight \
--labelled_idx $FACTORS \
--batch_size 24 \
--evaluate_iter 1000 \
--evaluation_metric sap_score accuracy \
--test_output_dir=$OUTPUT/test_output \
--train_output_dir=$OUTPUT/train_output \
--ckpt_dir=$OUTPUT/checkpoints \
--latent_reg 0.5


