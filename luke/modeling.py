import paddle
import paddle.nn as nn
from paddle.nn import Layer
from paddlenlp.transformers import register_base_model
from paddlenlp.transformers.model_utils import PretrainedModel

from luke.encoder import LukeEncoder


__all__ = [
    'LukePreTrainedModel',
    'LukeModel',
    'LukeForEntityClassification',
]


class LukePreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def forward(self, *inputs, **kwargs):
        pass

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "luke": {
            "vocab_size": 50267,
            "entity_vocab_size": 500000,
            "hidden_size": 768,
            "entity_emb_size": 256,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "use_entity_aware_attention": True,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "luke"

    def init_weights(self, module: Layer):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.set_value(
                paddle.normal(mean=0.0, std=self.initializer_range, shape=module.weight.shape)
            )
            if module.bias is not None:
                module.bias.set_value(
                    paddle.zeros(shape=module.bias.shape, dtype=module.bias.dtype)
                )
        elif isinstance(module, nn.Embedding):
            if module.__getattr__("_embedding_dim") == 1:  # embedding for bias parameters
                module.bias.set_value(
                    paddle.zeros(shape=module.bias.shape, dtype=module.bias.dtype)
                )
            else:
                module.weight.set_value(
                    paddle.normal(mean=0.0, std=self.initializer_range, shape=module.weight.shape)
                )
            padding_idx = module.__getattr__("_padding_idx")
            if padding_idx is not None:
                module.weight[padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.set_value(
                paddle.zeros(shape=module.bias.shape, dtype=module.bias.dtype)
            )
            module.weight.set_value(
                paddle.full(shape=module.weight.shape, fill_value=1.0, dtype=module.weight.dtype)
            )


@register_base_model
class LukeModel(LukePreTrainedModel):
    """
    "The bare LUKE model transformer outputting raw hidden-states for both word tokens and entities
    without any specific head on top.",
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self,
                 vocab_size=50267,
                 entity_vocab_size=500000,
                 hidden_size=768,
                 entity_emb_size=256,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 use_entity_aware_attention=True,
                 pad_token_id=1,
                 bos_token_id=0,
                 eos_token_id=2,
                 add_pooling_layer=True):
        super().__init__()
        self.initializer_range = initializer_range

        self.embeddings = LukeEmbeddings(vocab_size,
                                         hidden_size,
                                         pad_token_id,
                                         max_position_embeddings,
                                         type_vocab_size,
                                         layer_norm_eps,
                                         hidden_dropout_prob)
        self.entity_embeddings = LukeEntityEmbeddings(entity_vocab_size,
                                                      entity_emb_size,
                                                      hidden_size,
                                                      max_position_embeddings,
                                                      type_vocab_size,
                                                      layer_norm_eps,
                                                      hidden_dropout_prob)

        self.encoder = LukeEncoder(
            hidden_size,
            num_attention_heads,
            use_entity_aware_attention,
            attention_probs_dropout_prob,
            layer_norm_eps,
            hidden_dropout_prob,
            intermediate_size,
            hidden_act,
            num_hidden_layers
        )

        self.pooler = LukePooler(hidden_size) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_entity_embeddings(self):
        return self.entity_embeddings.entity_embeddings

    def set_entity_embeddings(self, value):
        self.entity_embeddings.entity_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            entity_ids=None,
            entity_attention_mask=None,
            entity_token_type_ids=None,
            entity_position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = paddle.ones((batch_size, seq_length))
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        if entity_ids is not None:
            entity_seq_length = entity_ids.shape[1]
            if entity_attention_mask is None:
                entity_attention_mask = paddle.ones((batch_size, entity_seq_length))
            if entity_token_type_ids is None:
                entity_token_type_ids = paddle.zeros((batch_size, entity_seq_length), dtype="int64")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # First, compute word embeddings
        word_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Second, compute extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, entity_attention_mask)

        # Third, compute entity embeddings and concatenate with word embeddings
        if entity_ids is None:
            entity_embedding_output = None
        else:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_token_type_ids)

        # Fourth, send embeddings through the model
        encoder_outputs = self.encoder(
            word_embedding_output,
            entity_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Fifth, get the output. LukeModel outputs the same as BertModel, namely sequence_output of shape (batch_size, seq_len, hidden_size)
        if return_dict:
            sequence_output = encoder_outputs["last_hidden_state"]
        else:
            sequence_output = encoder_outputs[0]

        # Sixth, we compute the pooled_output, word_sequence_output and entity_sequence_output based on the sequence_output
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseLukeModelOutputWithPooling(
        return dict(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs["hidden_states"],
            attentions=encoder_outputs["attentions"],
            entity_last_hidden_state=encoder_outputs["entity_last_hidden_state"],
            entity_hidden_states=encoder_outputs["entity_hidden_states"],
        )

    def get_extended_attention_mask(
            self, word_attention_mask, entity_attention_mask
    ):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        """
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = paddle.concat([attention_mask, entity_attention_mask], axis=-1)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = extended_attention_mask.astype(self._dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class LukeForEntityClassification(LukePreTrainedModel):
    """
    The LUKE model with a classification head on top (a linear layer on top of the hidden state of the first entity
    token) for entity classification tasks, such as Open Entity.
    """

    def __init__(self,
                 luke: LukeModel,
                 hidden_size=1024,
                 num_labels=9,
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02
                 ):
        super().__init__()
        self.initializer_range = initializer_range

        self.luke = luke

        self.num_labels = num_labels
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights and apply final processing
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            entity_ids=None,
            entity_attention_mask=None,
            entity_token_type_ids=None,
            entity_position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)` or :obj:`(batch_size, num_labels)`, `optional`):
            Labels for computing the classification loss. If the shape is :obj:`(batch_size,)`, the cross entropy loss
            is used for the single-label classification. In this case, labels should contain the indices that should be
            in :obj:`[0, ..., config.num_labels - 1]`. If the shape is :obj:`(batch_size, num_labels)`, the binary
            cross entropy loss is used for the multi-label classification. In this case, labels should only contain
            ``[0, 1]``, where 0 and 1 indicate false and true, respectively.

        Returns:
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        feature_vector = outputs["entity_last_hidden_state"][:, 0, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            # When the number of dimension of `labels` is 1, cross entropy is used as the loss function. The binary
            # cross entropy is used otherwise.
            if labels.ndim == 1:
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits))

        if not return_dict:
            output = (
                logits,
                outputs["hidden_states"],
                outputs["entity_hidden_states"],
                outputs["attentions"],
            )
            return ((loss,) + output) if loss is not None else output

        return dict(
            loss=loss,
            logits=logits,
            hidden_states=outputs["hidden_states"],
            entity_hidden_states=outputs["entity_hidden_states"],
            attentions=outputs["attentions"],
        )
        # return loss, logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions


class LukeEmbeddings(Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 pad_token_id,
                 max_position_embeddings,
                 type_vocab_size,
                 layer_norm_eps,
                 hidden_dropout_prob
                 ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # End copy
        self.padding_idx = pad_token_id
        # #  中间过程对齐时将此段注释, 训练时加上
        # self.position_embeddings = nn.Embedding(
        #     max_position_embeddings, hidden_size, padding_idx=self.padding_idx
        # )

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = paddle.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype="int64"
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class LukeEntityEmbeddings(Layer):
    # def __init__(self, config: LukeConfig):
    def __init__(self, entity_vocab_size,
                 entity_emb_size,
                 hidden_size,
                 max_position_embeddings,
                 type_vocab_size,
                 layer_norm_eps,
                 hidden_dropout_prob):
        super().__init__()
        # self.config = config
        self.entity_emb_size = entity_emb_size
        self.hidden_size = hidden_size

        self.entity_embeddings = nn.Embedding(entity_vocab_size, entity_emb_size, padding_idx=0)
        if entity_emb_size != hidden_size:
            self.entity_embedding_dense = nn.Linear(entity_emb_size, hidden_size, bias_attr=False)

        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
            # self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
            self, entity_ids: paddle.Tensor, position_ids: paddle.Tensor, token_type_ids: paddle.Tensor = None
    ):
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.entity_emb_size != self.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clip(min=0))
        # position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embedding_mask = (position_ids != -1).astype(position_embeddings.dtype).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = paddle.sum(position_embeddings, axis=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(axis=-2).clip(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertPooler
class LukePooler(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    y = paddle.full(shape=input_ids.shape, fill_value=padding_idx, dtype="int64")
    # mask = input_ids.not_equal(padding_idx).int()
    mask = input_ids.not_equal(y)
    t = mask.astype("int32")
    t = paddle.cumsum(t, axis=1)
    # t = t.astype(mask.dtype)
    incremental_indices = t * mask
    res = incremental_indices.astype("int64") + padding_idx
    return res
