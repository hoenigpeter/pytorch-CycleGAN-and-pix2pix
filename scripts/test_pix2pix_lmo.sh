set -ex
python test.py --dataroot ./datasets/lmo/1 --name pix2pix_lmo_1 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
python test.py --dataroot ./datasets/lmo/5 --name pix2pix_lmo_5 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
python test.py --dataroot ./datasets/lmo/6 --name pix2pix_lmo_6 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
python test.py --dataroot ./datasets/lmo/8 --name pix2pix_lmo_8 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
python test.py --dataroot ./datasets/lmo/9 --name pix2pix_lmo_9 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
python test.py --dataroot ./datasets/lmo/10 --name pix2pix_lmo_10 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
python test.py --dataroot ./datasets/lmo/11 --name pix2pix_lmo_11 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch
python test.py --dataroot ./datasets/lmo/12 --name pix2pix_lmo_12 --model pix2pix --netG unet_128 --direction AtoB --dataset_mode aligned --norm batch

