set -ex
python train.py --dataroot ./datasets/lmo/5 --name pix2pix_lmo_5 --model pix2pix --netG unet_128 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
