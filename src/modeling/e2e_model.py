from src.modeling.modeling import ClipBertForPreTraining
from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch


class ClipBert(nn.Module):
    def __init__(self, config, transformer_cls=ClipBertForPreTraining):
        super(ClipBert, self).__init__()
        self.config = config
        self.transformer = transformer_cls(config)

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        visual_features = batch["visual_inputs"]
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, bert_weights_path=None):

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)
