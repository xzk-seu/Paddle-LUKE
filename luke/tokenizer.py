import itertools
import json
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer
# from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer


class Trie:
    def __init__(self):
        self.data = {}

    def add(self, word: str):
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str) -> List[str]:
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = None
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    lookahead_index = current
                    end = current
                    next_char = text[lookahead_index] if lookahead_index < len(text) else None
                    while next_char in trie_pointer:
                        trie_pointer = trie_pointer[next_char]
                        lookahead_index += 1
                        if "" in trie_pointer:
                            end = lookahead_index
                            skip = lookahead_index

                        if lookahead_index == len(text):
                            # End of string
                            break
                        next_char = text[lookahead_index]
                    # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


class LukeTokenizer(GPTTokenizer):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 entity_vocab_file,
                 task=None,
                 additional_special_tokens_ids=None,
                 max_entity_length=None,
                 max_mention_length=None,
                 add_prefix_space=True,
                 **kwargs):
        """

        vocab_file (str):
            file path of the vocabulary.
        encoder_json_path (str, optional):
            file path of the id to vocab.
        vocab_bpe_path (str, optional):
            file path of word merge text.
        """
        super().__init__(vocab_file,
                         # encoder_json_path=vocab_json_file,
                         # vocab_bpe_path=merges_file,
                         merges_file,
                         # unk_token="<unk>",
                         # sep_token="</s>",
                         pad_token="<pad>",
                         # cls_token="<s>",
                         # mask_token="<mask>",
                         # entity_token_1="<ent>",
                         # entity_token_2="<ent2>",
                         # **kwargs
                         )
        self.unique_no_split_tokens = ['</s>', '<ent2>', '<ent>', '<mask>', '<pad>', '<s>', '<unk>']
        self.special_tokens_map = {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>',
                                   'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>',
                                   'additional_special_tokens': ['<ent>', '<ent2>']}
        self.task = task
        self.max_entity_length = max_entity_length
        self.max_mention_length = max_mention_length
        self.add_prefix_space = add_prefix_space

        self.tokens_trie = Trie()
        self._create_trie(self.unique_no_split_tokens)
        # self.set_special_tokens()
        # self.all_special_tokens = ['<ent>', '<ent2>']
        self.additional_special_tokens_ids = additional_special_tokens_ids

        if vocab_file[:-10] != entity_vocab_file[:-17]:
            entity_vocab_file = vocab_file[:-10] + entity_vocab_file[-17:]
        with open(entity_vocab_file, encoding="utf-8") as entity_vocab_handle:
            self.entity_vocab = json.load(entity_vocab_handle)

    def __call__(self,
                 text,
                 text_pair=None,
                 entities=None,
                 entities_pair=None,
                 entity_spans=None,
                 entity_spans_pair=None,
                 *args,
                 **kwargs):
        is_valid_single_text = isinstance(text, str)
        is_valid_batch_text = isinstance(text, (list, tuple)) and (len(text) == 0 or (isinstance(text[0], str)))
        assert (
                is_valid_single_text or is_valid_batch_text
        ), "text input must be of type `str` (single example) or `List[str]` (batch)."

        is_batched = bool(isinstance(text, (list, tuple)))

        if is_batched:
            batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
            if entities is None:
                batch_entities_or_entities_pairs = None
            else:
                batch_entities_or_entities_pairs = (
                    list(zip(entities, entities_pair)) if entities_pair is not None else entities
                )

            if entity_spans is None:
                batch_entity_spans_or_entity_spans_pairs = None
            else:
                batch_entity_spans_or_entity_spans_pairs = (
                    list(zip(entity_spans, entity_spans_pair)) if entity_spans_pair is not None else entity_spans
                )

            return self.batch_encode_plus(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                batch_entity_spans_or_entity_spans_pairs=batch_entity_spans_or_entity_spans_pairs,
                batch_entities_or_entities_pairs=batch_entities_or_entities_pairs,
                **kwargs,
            )
        else:
            return self._encode_plus(
                text=text,
                text_pair=text_pair,
                entity_spans=entity_spans,
                entity_spans_pair=entity_spans_pair,
                entities=entities,
                entities_pair=entities_pair,
                **kwargs,
            )

    def _encode_plus(
            self,
            text,
            text_pair=None,
            entities=None,
            entities_pair=None,
            entity_spans=None,
            entity_spans_pair=None,
            **kwargs
    ):
        (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        ) = self._create_input_sequence(
            text=text,
            text_pair=text_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=entity_spans,
            entity_spans_pair=entity_spans_pair
        )

        # prepare_for_model will create the attention_mask and token_type_ids
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            entity_ids=first_entity_ids,
            pair_entity_ids=second_entity_ids,
            entity_token_spans=first_entity_token_spans,
            pair_entity_token_spans=second_entity_token_spans,
        )

    def tokenize(self, text, **kwargs) -> List[str]:
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = dict(
        )

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if right:
                    tokens[i + 1] = right.lstrip()
                if left:
                    tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _create_input_sequence(
            self,
            text,
            text_pair,
            entities,
            entities_pair,
            entity_spans,
            entity_spans_pair,
    ) -> Tuple[list, list, list, list, list, list]:
        def get_input_ids(t):
            tokens = self.tokenize(t)
            return self.convert_tokens_to_ids(tokens)

        def get_input_ids_and_entity_token_spans(t, ent_spans):
            if ent_spans is None:
                return get_input_ids(t), None

            cur = 0
            input_ids = []
            entity_token_spans = [None] * len(ent_spans)

            split_char_positions = sorted(frozenset(itertools.chain(*ent_spans)))
            char_pos2token_pos = {}

            for split_char_position in split_char_positions:
                orig_split_char_position = split_char_position
                if (
                    split_char_position > 0 and t[split_char_position - 1] == " "
                ):  # whitespace should be prepended to the following token
                    split_char_position -= 1
                if cur != split_char_position:
                    input_ids += get_input_ids(t[cur:split_char_position])
                    cur = split_char_position
                char_pos2token_pos[orig_split_char_position] = len(input_ids)

            input_ids += get_input_ids(t[cur:])

            entity_token_spans = [
                (char_pos2token_pos[char_start], char_pos2token_pos[char_end]) for char_start, char_end in ent_spans
            ]

            return input_ids, entity_token_spans

        first_ids, second_ids = None, None
        first_entity_ids, second_entity_ids = None, None
        first_entity_token_spans, second_entity_token_spans = None, None

        if self.task is None:
            unk_entity_id = self.entity_vocab["[UNK]"]
            mask_entity_id = self.entity_vocab["[MASK]"]

            if entity_spans is None:
                first_ids = get_input_ids(text)
            else:
                assert isinstance(entity_spans, list) and (
                    len(entity_spans) == 0 or isinstance(entity_spans[0], tuple)
                ), "entity_spans should be given as a list of tuples containing the start and end character indices"
                assert entities is None or (
                    isinstance(entities, list) and (len(entities) == 0 or isinstance(entities[0], str))
                ), "If you specify entities, they should be given as a list of entity names"
                assert entities is None or len(entities) == len(
                    entity_spans
                ), "If you specify entities, entities and entity_spans must be the same length"

                first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)
                if entities is None:
                    first_entity_ids = [mask_entity_id] * len(entity_spans)
                else:
                    first_entity_ids = [self.entity_vocab.get(entity, unk_entity_id) for entity in entities]

        elif self.task == "entity_classification":
            assert (
                isinstance(entity_spans, list) and len(entity_spans) == 1 and isinstance(entity_spans[0], tuple)
            ), "Entity spans should be a list containing " \
               "a single tuple containing the start and end character indices of an entity"

            first_entity_ids = [self.entity_vocab["[MASK]"]]
            first_ids, first_entity_token_spans = get_input_ids_and_entity_token_spans(text, entity_spans)

            # add special tokens to input ids
            entity_token_start, entity_token_end = first_entity_token_spans[0]
            first_ids = (
                first_ids[:entity_token_end] + [self.additional_special_tokens_ids[0]] + first_ids[entity_token_end:]
            )
            first_ids = (
                first_ids[:entity_token_start]
                + [self.additional_special_tokens_ids[0]]
                + first_ids[entity_token_start:]
            )
            first_entity_token_spans = [(entity_token_start, entity_token_end + 2)]

        else:
            raise ValueError(f"Task {self.task} not supported")

        return (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        )

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        entity_ids: Optional[List[int]] = None,
        pair_entity_ids: Optional[List[int]] = None,
        entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        pair_entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ):
        # Compute lengths
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        encoded_inputs = {}

        # Compute the total size of the returned word encodings
        # total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length and max_entity_length
        overflowing_tokens = []

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            entity_token_offset = 1  # 1 * <s> token
            pair_entity_token_offset = len(ids) + 3  # 1 * <s> token & 2 * <sep> tokens
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            entity_token_offset = 0
            pair_entity_token_offset = len(ids)

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Set max entity length
        # if not max_entity_length:
        #     max_entity_length = self.max_entity_length

        if entity_ids is not None:
            total_entity_len = 0
            num_invalid_entities = 0
            valid_entity_ids = [ent_id for ent_id, span in zip(entity_ids, entity_token_spans) if span[1] <= len(ids)]
            valid_entity_token_spans = [span for span in entity_token_spans if span[1] <= len(ids)]

            total_entity_len += len(valid_entity_ids)
            num_invalid_entities += len(entity_ids) - len(valid_entity_ids)

            valid_pair_entity_ids, valid_pair_entity_token_spans = None, None
            if pair_entity_ids is not None:
                valid_pair_entity_ids = [
                    ent_id
                    for ent_id, span in zip(pair_entity_ids, pair_entity_token_spans)
                    if span[1] <= len(pair_ids)
                ]
                valid_pair_entity_token_spans = [span for span in pair_entity_token_spans if span[1] <= len(pair_ids)]
                total_entity_len += len(valid_pair_entity_ids)
                num_invalid_entities += len(pair_entity_ids) - len(valid_pair_entity_ids)

            final_entity_ids = valid_entity_ids + valid_pair_entity_ids if valid_pair_entity_ids else valid_entity_ids
            encoded_inputs["entity_ids"] = list(final_entity_ids)
            entity_position_ids = []
            entity_start_positions = []
            entity_end_positions = []
            for (token_spans, offset) in (
                (valid_entity_token_spans, entity_token_offset),
                (valid_pair_entity_token_spans, pair_entity_token_offset),
            ):
                if token_spans is not None:
                    for start, end in token_spans:
                        start += offset
                        end += offset
                        position_ids = list(range(start, end))[: self.max_mention_length]
                        position_ids += [-1] * (self.max_mention_length - end + start)
                        entity_position_ids.append(position_ids)
                        entity_start_positions.append(start)
                        entity_end_positions.append(end - 1)

            encoded_inputs["entity_position_ids"] = entity_position_ids
            if self.task == "entity_span_classification":
                encoded_inputs["entity_start_positions"] = entity_start_positions
                encoded_inputs["entity_end_positions"] = entity_end_positions

            if return_token_type_ids:
                encoded_inputs["entity_token_type_ids"] = [0] * len(encoded_inputs["entity_ids"])

        # # Check lengths
        # self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
        encoded_inputs["entity_attention_mask"] = [1] * len(encoded_inputs["entity_ids"])

        return encoded_inputs

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return text, kwargs

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep
