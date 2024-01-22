set -ex
python test.py --dataroot ./datasets/lmo/5 --name pix2pix_lmo_5 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
