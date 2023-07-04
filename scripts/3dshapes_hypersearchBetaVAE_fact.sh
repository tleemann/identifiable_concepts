# Call 3dshapes_hypersearch[].sh <beta> <runid> <corr method> <corr sigma> <corrfeature1> <corrfeature2> <number corr features>
echo "Using beta=$1, corr=$4"
NAME="BetaVAE_Beta-$1_Corr-$4_R$2"
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
--filter_fn $3 \
--sigma_corr $4 \
--corr_feature1 $5 \
--corr_feature2 $6 \
--n_cor $7 \
--max_iter 300000 \
--batch_size 64 \
--all_iter 20000 \
--oversampling_factor 3 \
--evaluation_metric factor_vae_metric dci sap_score \
--test_output_dir=$OUTPUT/test_output \
--train_output_dir=$OUTPUT/train_output \
--ckpt_dir=$OUTPUT/checkpoints