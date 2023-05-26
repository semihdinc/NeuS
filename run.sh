# Download images from S3 using the given info file and ROI file. Downscale images and mask them wrt. ROI coordinates 
# scene="test_COXDEN001_ROI_1_min_elev"
# exp_name="exp_1k"

# # python ev-nerf-run.py   --scene $scene \
# #                         --exp_name $exp_name
                        
# python ev-nerf-run.py   --scene $scene \
#                         --exp_name $exp_name \
#                         --render_frames

# python3 exp_runner.py --mode train --case bmvs_stone --conf ./confs/evdata.conf
# python3 exp_runner.py --mode train --case bmvs_stone --conf ./confs/evdata.conf --is_continue
python3 exp_runner.py --mode validate_mesh --case bmvs_stone --conf ./confs/evdata.conf --is_continue