set -ex
python train.py --dataroot ./datasets/lmo/10 --name pix2pix_lmo_10 --model pix2pix --netG unet_128 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
