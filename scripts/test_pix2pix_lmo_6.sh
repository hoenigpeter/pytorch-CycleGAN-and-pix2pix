set -ex
python test.py --dataroot ./datasets/lmo/6 --name pix2pix_lmo_6 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
