# Call 3dshapes_hypersearch[].sh <beta> <runid> <corr method> <corr sigma> <corrfeature1> <corrfeature2> <number corr features>
echo "Using lambda_od=$1, corr=$4"
NAME="DIPVAE_lambda-$1_Corr-$4_R$2"
python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--loss_terms=DIPVAEI \
--dset_dir=$DSET_PATH  \
--dset_name=shapes3d \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--z_dim=6 \
--use_wandb=false \
--w_kld=1.0 \
--w_dipvae=1.5 \
--lambda_od $1 \
--filter_fn $3 \
--sigma_corr $4 \
--corr_feature1 $5 \
--corr_feature2 $6 \
--n_cor $7 \
--lr_G=0.0005 \
--evaluation_metric factor_vae_metric dci sap_score \
--max_iter 300000 \
--all_iter 20000 \
--oversampling_factor 3 \
--test_output_dir=$OUTPUT/test_output \
--train_output_dir=$OUTPUT/train_output \
--ckpt_dir=$OUTPUT/checkpoints

