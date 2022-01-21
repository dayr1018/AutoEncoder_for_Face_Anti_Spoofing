#python train_RGB.py --model dropout --message dropout_wo_low --lowdata false

#sleep(10)

python train_RGB.py --model dropout --message dropout_w_low --lowdata true

python train_RGB_Depth.py --model depth_v1 --message depth_v1_w_low --lowdata true
