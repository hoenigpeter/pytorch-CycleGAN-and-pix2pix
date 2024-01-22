set -ex
python test.py --dataroot ./datasets/lmo --name lmo_pix2pix --model pix2pix --netG unet_256 --direction AtoB --dataset_mode aligned --norm batch
