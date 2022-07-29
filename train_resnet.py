import os
os.environ['TL_BACKEND'] = 'paddle'

from tlxzoo.datasets import DataLoaders
from tlxzoo.module.efficientnet import EfficientnetTransform
from tlxzoo.vision.image_classification import ImageClassification
import tensorlayerx as tlx

import paddle
import paddle.distributed as dist

if __name__ == '__main__':
    imagenet = DataLoaders('Imagenet', root_path='/home/vipuser/docker_root/imagenet_dataset',
                           per_device_train_batch_size=64, per_device_eval_batch_size=32)
    transform = EfficientnetTransform('efficientnet_b0')

    imagenet.register_transform_hook(transform)

    dist.init_parallel_env()
    model = ImageClassification(backbone="resnet50", num_labels=1000, input_shape=(1, 32, 32, 3))

    optimizer = tlx.optimizers.Adam(0.00001)
    metric = tlx.metrics.Accuracy()

    n_epoch = 800

    trainer = tlx.model.Model(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metric)
    trainer.train(n_epoch=n_epoch, train_dataset=imagenet.train, test_dataset=imagenet.test, print_freq=1,
                  print_train_batch=False)

    model.save_weights("./model.npz")

