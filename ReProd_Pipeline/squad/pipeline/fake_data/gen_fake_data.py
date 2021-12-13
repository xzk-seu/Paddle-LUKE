import json

import numpy as np
import paddle
from transformers import LukeTokenizer
import torch
from task.squad.train import load_examples, get_args_parser
from luke.tokenizer import LukeTokenizer
np.random.seed(42)


def gen_fake_data():
    """
    random.randint(low, high=None, size=None, dtype=int)
    :return:
    :rtype:
    """
    args = get_args_parser().parse_args()
    args.tokenizer = LukeTokenizer.from_pretrained("weight/pd/luke-large-finetuned-open-entity")
    train_dataloader, examples, features, _ = load_examples(args, is_evaluate=True)
    e = examples[0]
    f = features[0]
    example = dict(
        answer_texts=e.answer_texts,
        answers=e.answers,
        context_text=e.context_text,
        doc_tokens=e.doc_tokens,
        end_positions=e.end_positions,
        is_impossible=e.is_impossible,
        qas_id=e.qas_id,
        question_text=e.question_text,
        start_positions=e.start_positions,
        title=e.title
    )
    feature=dict(
        token_is_max_context=f.token_is_max_context,
        token_to_orig_map=f.token_to_orig_map,
        tokens=f.tokens,
        word_ids=f.word_ids,
        word_attention_mask=f.word_attention_mask,
        word_segment_ids=f.word_segment_ids
    )
    json.dump(example, open("example.json", "w"))
    json.dump(feature, open("feature.json", "w"))
    for i in train_dataloader:
        torch_input_dict = i
        break

    fake_data = {k: v.detach().numpy() for k, v in torch_input_dict.items()}
    # fake_data = [t.detach().numpy() for t in torch_input_list]

    np.savez("ReProd_Pipeline/squad/pipeline/fake_data/fake_data.npz", **fake_data)


if __name__ == "__main__":
    gen_fake_data()
