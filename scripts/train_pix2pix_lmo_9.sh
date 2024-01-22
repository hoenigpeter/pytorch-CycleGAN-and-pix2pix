set -ex
python train.py --dataroot ./datasets/lmo/9 --name pix2pix_lmo_9 --model pix2pix --netG unet_128 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
