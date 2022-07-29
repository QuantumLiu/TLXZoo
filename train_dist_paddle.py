import os
os.environ['TL_BACKEND'] = 'paddle'

import tensorlayerx as tlx
from tlxzoo.datasets import DataLoaders
from tlxzoo.module.efficientnet import EfficientnetTransform
from tlxzoo.vision.image_classification import ImageClassification

import paddle
from paddle.distributed import fleet
# 1.开启动态图模式
paddle.disable_static()

# 分布式step 2: 初始化fleet
fleet.init(is_collective=True)

if __name__ == '__main__':
    transform = EfficientnetTransform('efficientnet_b0')
    imagenet = DataLoaders('Imagenet', root_path='/home/vipuser/docker_root/imagenet_dataset',
                           per_device_train_batch_size=256, per_device_eval_batch_size=32)
    imagenet.register_transform_hook(transform)

    model = ImageClassification(backbone='resnet50', num_labels=1000)

    optimizer = tlx.optimizers.RMSprop(0.0001, momentum=0.9, weight_decay=1e-5)
    metric = tlx.metrics.Accuracy()

# 通过Fleet API获取分布式model和optimizer，用于支持分布式训练
    strategy = fleet.DistributedStrategy()

    optimizer = fleet.distributed_optimizer(optimizer)

    n_epoch = 300

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metric=metric)
    trainer.train(n_epoch=n_epoch, train_dataset=imagenet.train, test_dataset=imagenet.test, print_freq=1,
                  print_train_batch=False)

    model.save_weights('./model_resnet50.npz')
