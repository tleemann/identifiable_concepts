# Call 3dshapes_hypersearch[].sh <beta> <corr sigma> <corrfeature1> <corrfeature2> <runid>
echo "Using beta=$1, corr=$2"
if [ -z ${5+x} ]; then RUNID="0"; else RUNID=$5; fi # is arg 5 set?
NAME="BetaVAE_Beta-$1_Corr-$2_R${RUNID}"
echo $NAME
python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--dset_dir=$DSET_PATH  \
--dset_name=shapes3d \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=6 \
--w_kld=$1 \
--use_wandb=false \
--sigma_corr $2 \
--corr_feature1 $3 \
--corr_feature2 $4 \
--max_iter 300000 \
--batch_size 64 \
--all_iter 20000 \
--oversampling_factor 3 \
--evaluation_metric factor_vae_metric dci sap_score \
--test_output_dir=$OUTPUT/test_output \
--train_output_dir=$OUTPUT/train_output \
--ckpt_dir=$OUTPUT/checkpoints