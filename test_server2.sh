python test_RGB.py --checkpoint original_w_low --datatype 0 --lowdata true --threshold 460 --message original_wlow_0_460
python test_RGB.py --checkpoint dropout_w_low --datatype 0 --lowdata true --threshold 290 --message dropout_wlow_0_290
python test_RGB_Depth.py --checkpoint depth_v1_w_low --datatype 0 --lowdata true --threshold 7950 --message depth_v1_wlow_0_7950
python test_RGB_Depth.py --checkpoint depth_v2_w_low --datatype 0 --lowdata true --threshold 270 --message depth_v2_wlow_0_270

python test_RGB.py --checkpoint original_wo_low --datatype 0 --lowdata false --threshold 5750 --message original_wlow_0_5750
python test_RGB.py --checkpoint dropout_wo_low --datatype 0 --lowdata false --threshold 370 --message dropout_wlow_0_370
python test_RGB_Depth.py --checkpoint depth_v1_wo_low --datatype 0 --lowdata false --threshold 7560 --message depth_v1_wlow_0_7560
python test_RGB_Depth.py --checkpoint depth_v2_wo_low --datatype 0 --lowdata false --threshold 200 --message depth_v2_wlow_0_200

python test_RGB.py --checkpoint original_w_low --datatype 1 --lowdata true --threshold 310 --message original_wlow_1_310
python test_RGB.py --checkpoint dropout_w_low --datatype 1 --lowdata true --threshold 50 --message dropout_wlow_1_50
python test_RGB_Depth.py --checkpoint depth_v1_w_low --datatype 1 --lowdata true --threshold 110 --message depth_v1_wlow_1_110
python test_RGB_Depth.py --checkpoint depth_v2_w_low --datatype 1 --lowdata true --threshold 50 --message depth_v2_wlow_1_50

python test_RGB.py --checkpoint original_wo_low --datatype 1 --lowdata false --threshold 3450 --message original_wolow_1_3450
python test_RGB.py --checkpoint dropout_wo_low --datatype 1 --lowdata false --threshold 240 --message dropout_wolow_1_240
python test_RGB_Depth.py --checkpoint depth_v1_wo_low --datatype 1 --lowdata false --threshold 240 --message depth_v1_wolow_1_240
python test_RGB_Depth.py --checkpoint depth_v2_wo_low --datatype 1 --lowdata false --threshold 200 --message depth_v2_wolow_1_200
