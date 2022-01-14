# Charon's Code

## Input

1. 固定了细胞种类的stratified分fold快速脚本

## Utils

1. aug.py内置自定义数据增强的DatasetMapper
2. post_processing含有一个计算iou的函数，原本是用在投票法模型融合上的，但发现该方法并不有效.
3. add_swin_config是用来实现swin-transformer作为backbone的一个config修改函数 具体调用在train_swin_T.ipynb中

## Model

1. 实现swin-transformer作为backbone的额外文件例如detectron版本的config和swin-transformer本身的实现文件，用来在detectron中register该backbone(detectron中register组件是以函数的形式进行的，故需要一个register_swint_fpn进行swin-transformer类的调用)

## Ipynb
1. cascade_finetune.ipynb用于cascade模型的finetune
2. main_launch.ipynb是基础的模型训练文件
3. train_swin_T.ipynb用于迁徙swin-transformer作为backbone的mask_rcnn_r_50的训练
4. validation_v0.ipynb用于线下的交叉验证以及调参测试各种后处理手段
5. 其他几个ipynb后续有空补上介绍 大致作用是模型融合后验证 模型预训练 以及最终模型融合策略，线上融合提交
