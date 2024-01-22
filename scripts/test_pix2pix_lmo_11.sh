set -ex
python test.py --dataroot ./datasets/lmo/11 --name pix2pix_lmo_11 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
