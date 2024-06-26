CUDA_VISIBLE_DEVICES=0 python main.py \
    --device cuda \
    --dataset CIFAR10 --protocal I \
    --notes "cifar10_i/32bits" \
    --trainable_layer_num 2 \
    --M 4 \
    --feat_dim 64 \
    --T 0.2 \
    --hp_beta 5e-3 \
    --softmax_temp 5.0 \
    --clip_r 0.6 \
    --init_neg_curvs 1.0 \
    --full_hyperpq \
    --disable_writer \
    --epoch_num 50 \
    --clus_mode "hier_clus" \
    --num_clus_list "200,100,50" \
    --warmup_epoch 5 \
    --eval_interval 2 \
    --topK 1000 