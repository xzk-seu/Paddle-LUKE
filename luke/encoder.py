import math

import paddle
import paddle.nn as nn
from paddle.nn import Layer


class LukeLayer(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 use_entity_aware_attention,
                 attention_probs_dropout_prob,
                 layer_norm_eps,
                 hidden_dropout_prob,
                 intermediate_size,
                 hidden_act
                 ):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = LukeAttention(
            hidden_size,
            num_attention_heads,
            use_entity_aware_attention,
            attention_probs_dropout_prob,
            layer_norm_eps,
            hidden_dropout_prob
        )
        self.intermediate = LukeIntermediate(
            hidden_size,
            intermediate_size,
            hidden_act
        )
        self.output = LukeOutput(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            layer_norm_eps
        )

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        word_size = word_hidden_states.shape[1]

        self_attention_outputs = self.attention(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        if entity_hidden_states is None:
            concat_attention_output = self_attention_outputs[0]
        else:
            concat_attention_output = paddle.concat(self_attention_outputs[:2], axis=1)

        outputs = self_attention_outputs[2:]  # add self attentions if we output attention weights

        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, concat_attention_output
        # )
        intermediate_output = self.intermediate(concat_attention_output)
        layer_output = self.output(intermediate_output, concat_attention_output)

        word_layer_output = layer_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_layer_output = None
        else:
            entity_layer_output = layer_output[:, word_size:, :]

        outputs = (word_layer_output, entity_layer_output) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LukeEncoder(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 use_entity_aware_attention,
                 attention_probs_dropout_prob,
                 layer_norm_eps,
                 hidden_dropout_prob,
                 intermediate_size,
                 hidden_act,
                 num_hidden_layers
                 ):
        super().__init__()

        self.layer = nn.LayerList([LukeLayer(
            hidden_size,
            num_attention_heads,
            use_entity_aware_attention,
            attention_probs_dropout_prob,
            layer_norm_eps,
            hidden_dropout_prob,
            intermediate_size,
            hidden_act
        ) for _ in range(num_hidden_layers)])

        # self.layer = nn.TransformerEncoder(single_layer, num_hidden_layers)
        self.gradient_checkpointing = False

    def forward(
        self,
        word_hidden_states,
        entity_hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_word_hidden_states = () if output_hidden_states else None
        all_entity_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                word_hidden_states,
                entity_hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )

            word_hidden_states = layer_outputs[0]

            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    word_hidden_states,
                    all_word_hidden_states,
                    all_self_attentions,
                    entity_hidden_states,
                    all_entity_hidden_states,
                ]
                if v is not None
            )
        return dict(
            last_hidden_state=word_hidden_states,
            hidden_states=all_word_hidden_states,
            attentions=all_self_attentions,
            entity_last_hidden_state=entity_hidden_states,
            entity_hidden_states=all_entity_hidden_states,
        )


class LukeAttention(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 use_entity_aware_attention,
                 attention_probs_dropout_prob,
                 layer_norm_eps,
                 hidden_dropout_prob
                 ):
        super().__init__()
        self.self = LukeSelfAttention(
            hidden_size,
            num_attention_heads,
            use_entity_aware_attention,
            attention_probs_dropout_prob,
            embedding_size=None
        )
        self.output = LukeSelfOutput(
            hidden_size,
            layer_norm_eps,
            hidden_dropout_prob
        )
        self.pruned_heads = set()

    def forward(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        word_size = word_hidden_states.shape[1]  # .size(1)
        self_outputs = self.self(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        if entity_hidden_states is None:
            concat_self_outputs = self_outputs[0]
            concat_hidden_states = word_hidden_states
        else:
            concat_self_outputs = paddle.concat(self_outputs[:2], axis=1)
            concat_hidden_states = paddle.concat([word_hidden_states, entity_hidden_states], axis=1)

        attention_output = self.output(concat_self_outputs, concat_hidden_states)

        word_attention_output = attention_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_attention_output = None
        else:
            entity_attention_output = attention_output[:, word_size:, :]

        # add attentions if we output them
        outputs = (word_attention_output, entity_attention_output) + self_outputs[2:]

        return outputs


class LukeIntermediate(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act
                 ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            """
            ACT2FN是激活函数的字典，它的代码如下：
            ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
            # self.intermediate_act_fn = ACT2FN[config.hidden_act]
            """
            ACT2FN = {"gelu": paddle.nn.functional.gelu,
                      "relu": paddle.nn.functional.relu,
                      "swish": paddle.nn.functional.swish}
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LukeOutput(Layer):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_dropout_prob,
                 layer_norm_eps
                 ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LukeSelfAttention(Layer):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 use_entity_aware_attention,
                 attention_probs_dropout_prob,
                 embedding_size=None
                 ):
        super().__init__()
        # if hidden_size % num_attention_heads != 0 and not hasattr(config, "embedding_size"):
        if hidden_size % num_attention_heads != 0 and not embedding_size:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_entity_aware_attention = use_entity_aware_attention

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        if self.use_entity_aware_attention:
            self.w2e_query = nn.Linear(hidden_size, self.all_head_size)
            self.e2w_query = nn.Linear(hidden_size, self.all_head_size)
            self.e2e_query = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        # x = x.view(*new_x_shape)
        x = paddle.reshape(x, new_x_shape)
        x = paddle.transpose(x, (0, 2, 1, 3))
        return x

    def forward(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        word_size = word_hidden_states.shape[1]

        if entity_hidden_states is None:
            concat_hidden_states = word_hidden_states
        else:
            concat_hidden_states = paddle.concat([word_hidden_states, entity_hidden_states], axis=1)

        key_layer = self.transpose_for_scores(self.key(concat_hidden_states))
        value_layer = self.transpose_for_scores(self.value(concat_hidden_states))

        if self.use_entity_aware_attention and entity_hidden_states is not None:
            # compute query vectors using word-word (w2w), word-entity (w2e), entity-word (e2w), entity-entity (e2e)
            # query layers
            w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

            # compute w2w, w2e, e2w, and e2e key vectors used with the query vectors computed above
            w2w_key_layer = key_layer[:, :, :word_size, :]
            e2w_key_layer = key_layer[:, :, :word_size, :]
            w2e_key_layer = key_layer[:, :, word_size:, :]
            e2e_key_layer = key_layer[:, :, word_size:, :]

            # compute attention scores based on the dot product between the query and key vectors
            perm = [i for i in range(len(w2w_key_layer.shape))]
            perm[-1], perm[-2] = perm[-2], perm[-1]
            w2w_attention_scores = paddle.matmul(w2w_query_layer, w2w_key_layer.transpose(perm))
            w2e_attention_scores = paddle.matmul(w2e_query_layer, w2e_key_layer.transpose(perm))
            e2w_attention_scores = paddle.matmul(e2w_query_layer, e2w_key_layer.transpose(perm))
            e2e_attention_scores = paddle.matmul(e2e_query_layer, e2e_key_layer.transpose(perm))

            # combine attention scores to create the final attention score matrix
            word_attention_scores = paddle.concat([w2w_attention_scores, w2e_attention_scores], axis=3)
            entity_attention_scores = paddle.concat([e2w_attention_scores, e2e_attention_scores], axis=3)
            attention_scores = paddle.concat([word_attention_scores, entity_attention_scores], axis=2)

        else:
            query_layer = self.transpose_for_scores(self.query(concat_hidden_states))
            attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in LukeModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size,]
        context_layer = context_layer.reshape(new_context_layer_shape)

        output_word_hidden_states = context_layer[:, :word_size, :]
        if entity_hidden_states is None:
            output_entity_hidden_states = None
        else:
            output_entity_hidden_states = context_layer[:, word_size:, :]

        if output_attentions:
            outputs = (output_word_hidden_states, output_entity_hidden_states, attention_probs)
        else:
            outputs = (output_word_hidden_states, output_entity_hidden_states)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class LukeSelfOutput(Layer):
    def __init__(self,
                 hidden_size,
                 layer_norm_eps,
                 hidden_dropout_prob
                 ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
