#! /bin/bash

# # full_epnet_without_iou_branch
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_without_iou_branch/eval_results/  --ckpt ./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth --set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# # epnet_without_ce_loss
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/epnet_without_ce_loss/eval_results/  --ckpt ./log/Car/models/epnet_without_ce_loss/ckpt/checkpoint_epoch_44.pth --set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# # epnet_without_AF_Fusion
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online   --output_dir ./log/Car/models/epnet_without_AF_Fusion/eval_results/  --ckpt ./log/Car/models/epnet_without_AF_Fusion/ckpt/checkpoint_epoch_44.pth --set  AF_FUSION.ENABLED False AF_FUSION.ADD_Image_Attention False RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# # pointrcnn_ori
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online   --output_dir ./log/Car/models/pointrcnn_ori/eval_results/  --ckpt ./log/Car/models/pointrcnn_ori/ckpt/checkpoint_epoch_47.pth --set  AF_FUSION.ENABLED False AF_FUSION.ADD_Image_Attention False RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# # full_epnet_with_iou_branch
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_with_iou_branch/eval_results/  --ckpt ./log/Car/models/full_epnet_with_iou_branch/ckpt/checkpoint_epoch_46.pth --set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH True
# full_epnet_without_iou_branch
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml --eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_without_iou_branch211028/eval_results1028/  --ckpt /home/bichunyu/Documents/project/CYNet2/EPNet/tools/log/Car/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_20.pth --set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py \
# 					--cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml \
# 					--eval_mode rcnn_online \
# 					--output_dir ./log/Car/AF_POINT_211104/without_CE_exp1_batchsize2/eval \
# 					--ckpt /home/bichunyu/Documents/project/CYNet2/EPNet/tools/log/Car/AF_POINT_211104/without_CE_exp1_batchsize2/ckpt/checkpoint_epoch_4.pth \
# 					--set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False 

# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py \
# 					--cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml \
# 					--eval_mode rcnn_online \
# 					--output_dir ./log/Car/AF2_POINT_211108/without_CE_exp1_batchsize2 \
# 					--ckpt /home/bichunyu/Documents/project/CYNet2/EPNet/tools/log/Car/AF2_POINT_211108/without_CE_exp1_batchsize2/ckpt/checkpoint_epoch_$1.pth \
# 					--set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py \
# 					--cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml \
# 					--eval_mode rcnn_online \
# 					--output_dir ./logs/Perdiam/AF2_POINT_0117/ \
# 					--ckpt ./logs/Perdiam/AF2_POINT_211230ms_abal/ckpt/checkpoint_epoch_$1.pth \
# 					--set  AF_FUSION.ENABLED False AF_FUSION.ADD_Image_Attention False RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py \
# 					--cfg_file cfgs/AF_Fusion_with_attention_use_ce_loss.yaml \
# 					--eval_mode rcnn_online \
# 					--output_dir ./logs/Perdiam/final/signle_branch \
# 					--ckpt ./logs/Perdiam/final/signle_branch/ckpt/checkpoint_epoch_$1.pth \
# 					--set  AF_FUSION.ENABLED True AF_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
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
