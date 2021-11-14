# NCE_ICC-2022
Source Codes for Neural Capacity Estimators: How reliable are they?

For producing results for various estimators copy and paste following command on the cmd of your local machine:

python pymain.py --SNR ''snr-value'' --init_epoch ''initial training phase iteration'' --max_epoch  '' joint training iterations'' --hidden_dim_critic '' hidden dimension of critic network (MI ESTIMATOR)'' --hidden_dim_nit ''hidden dimension of neural optimal input approximator'' --dim 1 --dim_nit 1 --layer_nit ''number of layers for neural optimal input approximator'' --layer_mi 'number of layers for MI Estimator networks' type_channel conts_awgn --peak ''peak constraints'' --positive ''for optical intensity channel'' --estimator "estimator-type"

Example : optical intensity at 8db : python pymain.py --SNR 8 --init_epoch 100 --max_epoch 1000 --hidden_dim_critic 256 --hidden_dim_nit 1024 --dim 1 --dim_nit 1 --layer_nit 4 --layer_mi 4 type_channel conts_awgn --peak 6.3095 --positive 1
