python valid_RGB.py --checkpoint original_w_low --datatype 0 --lowdata true --message t2_original_wlow_0
python valid_RGB.py --checkpoint dropout_w_low --datatype 0 --lowdata true --message t2_dropout_wlow_0
python valid_RGB_Depth.py --checkpoint depth_v1_w_low --datatype 0 --lowdata true --message t2_depthv1_wlow_0
python valid_RGB_Depth.py --checkpoint depth_v2_w_low --datatype 0 --lowdata true --message t2_depthv2_wlow_0

python valid_RGB.py --checkpoint original_wo_low --datatype 0 --lowdata false --message t2_original_wolow_0
python valid_RGB.py --checkpoint dropout_wo_low --datatype 0 --lowdata false --message t2_dropout_wolow_0
python valid_RGB_Depth.py --checkpoint depth_v1_wo_low --datatype 0 --lowdata false --message t2_depthv1_wolow_0
python valid_RGB_Depth.py --checkpoint depth_v2_wo_low --datatype 0 --lowdata false --message t2_depthv2_wolow_0

python valid_RGB.py --checkpoint original_w_low --datatype 1 --lowdata true --message t2_original_wlow_1
python valid_RGB.py --checkpoint dropout_w_low --datatype 1 --lowdata true --message t2_dropout_wlow_1
python valid_RGB_Depth.py --checkpoint depth_v1_w_low --datatype 1 --lowdata true --message t2_depthv1_wlow_1
python valid_RGB_Depth.py --checkpoint depth_v2_w_low --datatype 1 --lowdata true --message t2_depthv2_wlow_1

python valid_RGB.py --checkpoint original_wo_low --datatype 1 --lowdata false --message t2_original_wolow_1
python valid_RGB.py --checkpoint dropout_wo_low --datatype 1 --lowdata false --message t2_dropout_wolow_1
python valid_RGB_Depth.py --checkpoint depth_v1_wo_low --datatype 1 --lowdata false --message t2_depthv1_wolow_1
python valid_RGB_Depth.py --checkpoint depth_v2_wo_low --datatype 1 --lowdata false --message t2_depthv2_wolow_1

