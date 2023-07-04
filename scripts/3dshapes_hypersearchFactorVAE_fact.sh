# Call 3dshapes_hypersearch[].sh <beta> <runid> <corr method> <corr sigma> <corrfeature1> <corrfeature2> <number corr features>
echo "Using wtc=$1, corr=$4"
NAME="FactorVAE_wtc-$1_Corr-$4_R$2"
python3 main.py \
--name=$NAME \
--alg=BetaVAE \
--controlled_capacity_increase=true \
--loss_terms=FactorVAE \
--dset_dir=$DSET_PATH  \
--dset_name=shapes3d \
--traverse_z=true \
--encoder=SimpleGaussianConv64 \
--decoder=SimpleConv64 \
--discriminator=SimpleDiscriminator \
--z_dim=6 \
--w_kld=1 \
--w_tc=$1 \
--lr_G=0.002 \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_args mode=min factor=0.8 patience=0 min_lr=0.000001 \
--filter_fn $3 \
--sigma_corr $4 \
--corr_feature1 $5 \
--corr_feature2 $6 \
--n_cor $7 \
--max_iter 300000 \
--all_iter 20000 \
--oversampling_factor 3 \
--evaluation_metric factor_vae_metric dci sap_score \
--test_output_dir=$OUTPUT/test_output \
--train_output_dir=$OUTPUT/train_output \
--ckpt_dir=$OUTPUT/checkpoints
