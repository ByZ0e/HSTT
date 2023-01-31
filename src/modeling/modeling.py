"""
Transformer part of ClipBERT
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from .transformers import BertPreTrainedModel
from .transformers import (
    BertPreTrainingHeads, BertEmbeddings, BertEncoder, BertPooler)
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from src.utils.basic_utils import get_attention_mask, get_semantic_attention_mask, flat_list_of_lists
from src.utils.load_save import load_state_dict_with_mismatch
import pdb

def get_random_sample_indices(
        seq_len, num_samples=100, device=torch.device("cpu")):
    """
    Args:
        seq_len: int, the sampled indices will be in the range [0, seq_len-1]
        num_samples: sample size
        device: torch.device

    Returns:
        1D torch.LongTensor consisting of sorted sample indices
        (sort should not affect the results as we use transformers)
    """
    if num_samples >= seq_len:
        # return all indices
        sample_indices = np.arange(seq_len)
    else:
        sample_indices = np.random.choice(
            seq_len, size=num_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    return torch.from_numpy(sample_indices).long().to(device)


BertLayerNorm = LayerNorm


class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """
    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config
        self.num_sample = config.num_sample
        self.num_obj_per_frame = config.num_obj_per_frame
        self.num_rel_per_frame = config.num_rel_per_frame
        self.hidden_size = config.hidden_size
        # sequence embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.lin_obj = nn.Linear(config.obj_dim, config.hidden_size)
        self.lin_rel = nn.Linear(config.rel_dim, config.hidden_size)
        self.lin_frame = nn.Linear(config.frame_dim, config.hidden_size)
        self.lin_action = nn.Linear(config.action_dim, config.hidden_size)

        # self.padding = VisualTokenPadding(config.hidden_size)
        self.token_type_embeddings = nn.Embedding(5, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.temporal_embeddings = nn.Embedding(4000, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visual_inputs, frame_info):
        """
        Args:
            visual_inputs<list>: (B, dict)
            frame_info<list>: (B, list)
        Returns:

        """
        bsz = len(visual_inputs)

        d_comb = visual_inputs[0]
        for i in range(1, bsz):
            d_comb = {key: torch.cat((d_comb[key], visual_inputs[i][key])) for key in visual_inputs[0].keys()}
        f_obj = d_comb['object'].view(-1, self.config.obj_dim)
        f_rel = d_comb['relation'].view(-1, self.config.rel_dim)
        f_frame = d_comb['frame']
        f_action = d_comb['action']

        device = f_obj.device
        obj_token = self.lin_obj(f_obj)
        rel_token = self.lin_rel(f_rel)
        frame_token = self.lin_frame(f_frame)
        action_token = self.lin_action(f_action)

        num_objs, num_rels, num_frames, num_actions, num_tokens = [], [], [], [], []
        token_type_idxs = []
        temporal_idxs = []
        for info in frame_info:

            action_idxs, frame_idxs, rel_idxs = info

            num_obj = int(len(rel_idxs) / 2)
            num_rel = len(rel_idxs)
            num_frame = len(frame_idxs)
            num_action = len(action_idxs)
            # pdb.set_trace()
            num_tokens.append(num_obj + num_rel + num_frame + num_action)
            num_objs.append(num_obj)
            num_rels.append(num_rel)
            num_frames.append(num_frame)
            num_actions.append(num_action)

            token_type_idx = [
                torch.ones(num_obj, dtype=torch.long, device=device),
                2 * torch.ones(num_rel, dtype=torch.long, device=device),
                3 * torch.ones(num_frame, dtype=torch.long, device=device),
                4 * torch.ones(num_action, dtype=torch.long, device=device),
            ]
            token_type_idxs.append(self.token_type_embeddings(torch.cat(token_type_idx)))

        obj_token = obj_token.split(num_objs, dim=0)
        rel_token = rel_token.split(num_rels, dim=0)
        frame_token = frame_token.split(num_frames, dim=0)
        action_token = action_token.split(num_actions, dim=0)
        # pdb.set_trace()

        visual_tokens = [torch.cat((obj_token[i], rel_token[i], frame_token[i], action_token[i])) for i in range(0, bsz)]
        visual_tokens = nn.utils.rnn.pad_sequence(visual_tokens, batch_first=True)  # (B, Lv, d)
        token_type_embeddings = nn.utils.rnn.pad_sequence(token_type_idxs, batch_first=True)  # (B, Lv, d)
        # temporal_embeddings = nn.utils.rnn.pad_sequence(temporal_idxs, batch_first=True)  # (B, Lv, d)

        # -- Prepare masks
        pad_len = max(num_tokens)  # Lv
        num_tokens_ = torch.tensor(num_tokens, device=device).unsqueeze(1).expand(-1, pad_len)  # (bsz, pad_len)
        # slf_attn_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_tokens_).unsqueeze(1).expand(-1, pad_len, -1) # (bsz, pad_len, pad_len)
        non_pad_mask = torch.arange(pad_len, device=device).view(1, -1).expand(bsz, -1).lt(num_tokens_).squeeze(-1)  # (bsz, pad_len)

        position_ids = torch.arange(pad_len, dtype=torch.long,
                                    device=device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = visual_tokens + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, non_pad_mask  # (B, token_length, d)


class ClipBertBaseModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    config keys:
        text_model: str, text model name, default "bert-based-uncased"
        pretrained: bool, use pre-trained vision_model, default True
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.visual_embeddings = VisualInputEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_edge_mask_down(self, frame_info, text_len, visual_len):
        bsz = len(frame_info)
        # print(bsz, text_len, visual_len)
        pad_len = text_len + visual_len
        edge_mask = torch.ones(bsz, pad_len, pad_len, dtype=torch.long)
        for i, info in enumerate(frame_info):
            action_idxs, frame_idxs, rel_idxs = info

            num_obj = int(len(rel_idxs) / 2)
            num_rel = len(rel_idxs)
            num_frame = len(frame_idxs)
            num_action = len(action_idxs)
            num_token = num_obj + num_rel + num_frame + num_action

            # down
            edge_mask[i, text_len + num_obj:text_len + num_obj + num_rel, text_len:text_len + num_obj] = get_attention_mask(rel_idxs, num_obj).t()
            edge_mask[i, text_len + num_obj + num_rel:text_len + num_token - num_action, text_len + num_obj:text_len + num_obj + num_rel] = get_attention_mask(frame_idxs, num_rel).t()
            edge_mask[i, text_len + num_token - num_action:text_len + num_token, text_len + num_obj + num_rel:text_len + num_token - num_action] = get_attention_mask(action_idxs, num_frame).t()

            edge_mask[i, text_len + num_obj + num_rel:text_len + num_token, text_len:text_len + num_obj] = torch.zeros((num_frame+num_action, num_obj))
            edge_mask[i, text_len + num_token - num_action:text_len + num_token, text_len + num_obj:text_len + num_obj + num_rel] = torch.zeros((num_action, num_rel))

        return edge_mask

    def forward(self, text_input_ids, visual_inputs, frame_info, attention_mask):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B, #frame, H, W, C)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        input_shape = text_input_ids.size()
        device = text_input_ids.device
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        text_embedding_output = self.embeddings(
            input_ids=text_input_ids)  # (B, Lt, D)
        visual_embedding_output, non_pad_mask = self.visual_embeddings(
            visual_inputs, frame_info)  # (B, Lv, d)
        visual_attention_mask = attention_mask.new_ones(
            visual_embedding_output.shape[:2])  # (B, Lv)
        visual_attention_mask = visual_attention_mask * non_pad_mask
        edge_mask = self.get_edge_mask_down(frame_info, text_len=text_embedding_output.shape[1], visual_len=visual_embedding_output.shape[1])
        attention_mask = torch.cat(
            [attention_mask, visual_attention_mask], dim=-1)  # (B, lt+Lv, d)
        # pdb.set_trace()
        attention_mask = attention_mask[:, None, :] * edge_mask.to(device)
        embedding_output = torch.cat(
            [text_embedding_output, visual_embedding_output],
            dim=1)  # (B, Lt+Lv, d)
        extended_attention_mask: torch.Tensor =\
            self.get_extended_attention_mask(
                attention_mask, input_shape, device)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=self.get_head_mask(
                None, self.config.num_hidden_layers)  # required input
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class ClipBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        text_input_ids,
        visual_inputs,
        text_input_mask,
        mlm_labels=None,
        itm_labels=None,
    ):
        r"""
        text_input_ids: (B, Lt)
        visual_inputs: (B, #frame, H, W, C)
        text_input_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        mlm_labels: (B, Lt)
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        itm_label: (B, )  with 1 indicates positive pair, 0 indicates negative pair.
        """

        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )

        sequence_output, pooled_output = outputs[:2]
        # Only use the text part (which is the first `Lt` tokens) to save computation,
        # this won't cause any issue as cls only has linear layers.
        txt_len = text_input_mask.shape[1]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output[:, :txt_len], pooled_output)

        loss_fct = CrossEntropyLoss(reduction="none")
        if mlm_labels is not None:
            mlm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1))
        else:
            mlm_loss = 0
        if itm_labels is not None:
            itm_loss = loss_fct(
                seq_relationship_score.view(-1, 2), itm_labels.view(-1))
        else:
            itm_loss = 0

        return dict(
            mlm_scores=prediction_scores,  # (B, Lt, vocab_size),  only text part
            mlm_loss=mlm_loss,  # (B, )
            mlm_labels=mlm_labels,  # (B, Lt), with -100 indicates ignored positions
            itm_scores=seq_relationship_score,  # (B, 2)
            itm_loss=itm_loss,  # (B, )
            itm_labels=itm_labels  # (B, )
        )


def instance_bce_with_logits(logits, labels, reduction="mean"):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


ClipBertForSequenceClassificationConfig = dict(
    cls_hidden_scale=2,   # mlp intermediate layer hidden size scaler
    classifier="mlp",  # classfied type, [mlp, linear]
    num_labels=3129,  # number of labels for classifier output
    loss_type="bce"  # [BCE, CE, KLDivLoss] only used when num_labels > 1
)


class ClipBertForSequenceClassification(BertPreTrainedModel):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(ClipBertForSequenceClassification, self).__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

        self.init_weights()

    def forward(self, text_input_ids, visual_inputs, frame_info,
                text_input_mask, labels=None):
        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            frame_info=frame_info,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(
                    logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss

    def load_separate_ckpt(self, bert_weights_path=None):
        if bert_weights_path:
            load_state_dict_with_mismatch(self, bert_weights_path)


class ClipBertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(ClipBertForMultipleChoice, self).__init__(config)
        self.config = config

        self.bert = ClipBertBaseModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, 1)
        )
        self.init_weights()

    def forward(self, text_input_ids, visual_inputs, frame_info,
                text_input_mask, labels=None):
        """
        Args:
            text_input_ids: (B * num_labels, Lt)
            visual_inputs: (B, Lv, d)
            text_input_mask: (B * num_labels, Lt)
            labels: (B, ), in [0, num_labels-1]
        Returns:
        """
        outputs = self.bert(
            text_input_ids=text_input_ids,
            visual_inputs=visual_inputs,
            frame_info=frame_info,
            attention_mask=text_input_mask,  # (B, Lt) note this mask is text only!!!
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits, loss = self.calc_loss(logits, labels)
        return dict(
            logits=logits,
            loss=loss
        )

    def calc_loss(self, logits, labels):
        if self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
            logits = logits.view(-1, self.config.num_labels)

        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    # logits = logits.view(-1, self.config.num_labels)
                    loss = loss_fct(logits, labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states)
