#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py \
					--cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml \
					--eval_mode rcnn_online \
					--output_dir ./logs/test_infer_time \
					--ckpt ./logs/final.pth \
					--set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
# for ((i=0;i<50;i++))
# do
# 	CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./logs/Perdiam/AF2_POINT_0117/  --ckpt ./logs/Perdiam/AF2_POINT_211230ms_abal/ckpt/checkpoint_epoch_$i.pth --set  AF_FUSION.ENABLED False AF_FUSION.ADD_Image_Attention False RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
# done
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_with_iou_branch/eval_results/  --ckpt /home/bichunyu/Documents/project/CYNet2/EPNet/tools/log/Car/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_29.pth --set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True
