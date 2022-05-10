#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python train_rcnn.py \
                      --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml \
                      --batch_size 2 --train_mode rcnn_online \
                      --epochs 50 \
                      --ckpt_save_interval 5 \
                      --output_dir ./logs/Perdiam/final/signle_branch \
                      --set AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False TRAIN.CE_WEIGHT 5.0 
