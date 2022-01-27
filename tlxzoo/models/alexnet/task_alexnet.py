from ...task.task import BaseForImageClassification
from .alexnet import AlexNet
from ...utils.registry import Registers
from ...utils.output import BaseForImageClassificationTaskOutput
from .config_alexnet import AlexNetForImageClassificationTaskConfig
import tensorlayerx as tlx


@Registers.tasks.register
class AlexNetForImageClassification(BaseForImageClassification):
    config_class = AlexNetForImageClassificationTaskConfig

    def __init__(self,
                 config: AlexNetForImageClassificationTaskConfig,
                 model=None):
        super(AlexNetForImageClassification, self).__init__(config)

        if model is not None:
            self.alexnet = model
        else:
            self.alexnet = AlexNet(self.config.model_config)

        self.num_classes = config.num_classes
        try:
            in_channels = self.config.model_config.get_last_output_size()[-1]
            self.classifier = tlx.nn.Dense(n_units=self.num_classes,
                                           in_channels=in_channels,
                                           name="classifier")
        except:
            self.classifier = tlx.nn.Dense(n_units=self.num_classes,
                                           name="classifier")

    def forward(self, pixels, labels=None):
        outs = self.vgg(pixels)

        last_out = outs.output

        logits = self.classifier(last_out)
        if labels:
            loss = tlx.losses.softmax_cross_entropy_with_logits(labels, logits)
        else:
            loss = None

        return BaseForImageClassificationTaskOutput(logits=logits, loss=loss)
