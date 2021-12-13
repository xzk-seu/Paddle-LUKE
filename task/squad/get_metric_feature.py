import json
e2 = {'id': '56be4db0acb8001400a502ee', 'title': 'Super_Bowl_50',
      'context': 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.',
      'question': 'Where did Super Bowl 50 take place?',
      'answers': [
                      'Santa Clara, California',
                      "Levi's Stadium",
                      "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
      ],
      'answer_starts': [403, 355, 355], 'is_impossible': False
      }

f1 = {'input_ids': [101, 2029, 5088, 2136, 3421, 1996, 10511, 2012, 3565, 4605, 2753, 1029, 102, 3565, 4605, 2753, 2001, 2019, 2137, 2374, 2208, 2000, 5646, 1996, 3410, 1997, 1996, 2120, 2374, 2223, 1006, 5088, 1007, 2005, 1996, 2325, 2161, 1012, 1996, 2137, 2374, 3034, 1006, 10511, 1007, 3410, 7573, 14169, 3249, 1996, 2120, 2374, 3034, 1006, 22309, 1007, 3410, 3792, 12915, 2484, 1516, 2184, 2000, 7796, 2037, 2353, 3565, 4605, 2516, 1012, 1996, 2208, 2001, 2209, 2006, 2337, 1021, 1010, 2355, 1010, 2012, 11902, 1005, 1055, 3346, 1999, 1996, 2624, 3799, 3016, 2181, 2012, 4203, 10254, 1010, 2662, 1012, 2004, 2023, 2001, 1996, 12951, 3565, 4605, 1010, 1996, 2223, 13155, 1996, 1000, 3585, 5315, 1000, 2007, 2536, 2751, 1011, 11773, 11107, 1010, 2004, 2092, 2004, 8184, 28324, 2075, 1996, 102],
      # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      'offset_mapping': [None, None, None, None, None, None, None, None, None, None, None, None, None, (0, 5), (6, 10), (11, 13), (14, 17), (18, 20), (21, 29), (30, 38), (39, 43), (44, 46), (47, 56), (57, 60), (61, 69), (70, 72), (73, 76), (77, 85), (86, 94), (95, 101), (102, 103), (103, 106), (106, 107), (108, 111), (112, 115), (116, 120), (121, 127), (127, 128), (129, 132), (133, 141), (142, 150), (151, 161), (162, 163), (163, 166), (166, 167), (168, 176), (177, 183), (184, 191), (192, 200), (201, 204), (205, 213), (214, 222), (223, 233), (234, 235), (235, 238), (238, 239), (240, 248), (249, 257), (258, 266), (267, 269), (269, 270), (270, 272), (273, 275), (276, 280), (281, 286), (287, 292), (293, 298), (299, 303), (304, 309), (309, 310), (311, 314), (315, 319), (320, 323), (324, 330), (331, 333), (334, 342), (343, 344), (344, 345), (346, 350), (350, 351), (352, 354), (355, 359), (359, 360), (360, 361), (362, 369), (370, 372), (373, 376), (377, 380), (381, 390), (391, 394), (395, 399), (400, 402), (403, 408), (409, 414), (414, 415), (416, 426), (426, 427), (428, 430), (431, 435), (436, 439), (440, 443), (444, 448), (449, 454), (455, 459), (459, 460), (461, 464), (465, 471), (472, 482), (483, 486), (487, 488), (488, 494), (495, 506), (506, 507), (508, 512), (513, 520), (521, 525), (525, 526), (526, 532), (533, 544), (544, 545), (546, 548), (549, 553), (554, 556), (557, 568), (569, 576), (576, 579), (580, 583), (0, 0)],
      # 'overflow_to_sample': 0,
      'example_id': '56be4db0acb8001400a502ec'
      }


# def _get_offset_mapping(context, tokens, token_to_orig_map, doc_tokens):
#     offset_mapping = [None] * len(tokens)
#     offset_mapping[-1] = (0, 0)
#     s_idx = 0
#     e_idx = 0
#     for k, v in token_to_orig_map.items():
#         e_idx = s_idx + len(doc_tokens[v])
#         offset_mapping[k] = (s_idx, e_idx)


def _get_offset_mapping(context, tokens, token_to_orig_map, doc_tokens):
    offset_mapping = [None] * len(tokens)
    offset_mapping[-1] = (0, 0)
    flag = False
    idx = 0
    e_idx = 0
    for i in range(1, len(tokens)-1):
        if not flag:
            if tokens[i-1] == '</s>' and tokens[i] == '</s>':
                flag = True
                continue
            else:
                continue
        token = tokens[i].replace('Ġ', '')
        token = token.replace('âĢĵ', '–')
        token = token.replace('âĪĴ', '−')
        token = token.replace('Â·', '·')
        new_idx = context.find(token, e_idx)
        if new_idx == -1:
            raise Exception("_get_offset_mapping error")
        if e_idx+1 != new_idx and e_idx != new_idx:
            raise Exception("_get_offset_mapping idx error")
        idx = new_idx
        e_idx = idx + len(token)
        offset_mapping[i] = (idx, e_idx)

    return offset_mapping


def _get_example(ori_example):
    example = dict()
    example["id"] = ori_example["qas_id"]
    example["context"] = ori_example["context_text"]
    example["question"] = ori_example["question_text"]
    example["answers"] = ori_example["answer_texts"]
    example["answer_starts"] = [i['answer_start'] for i in ori_example["answers"]]
    example["is_impossible"] = ori_example["question_text"]
    return example


def _get_feature(ori_example, ori_feature):
    feature = dict()
    feature["input_ids"] = ori_feature["word_ids"]
    feature["example_id"] = ori_example["qas_id"]
    feature["offset_mapping"] = _get_offset_mapping(ori_example["context_text"], ori_feature["tokens"])
    return feature


def get_metric_feature(ori_example, ori_feature):
    if not (isinstance(ori_example, dict) and isinstance(ori_feature, dict)):
        e, f = ori_example, ori_feature
        ori_example = dict(
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
        ori_feature = dict(
            token_is_max_context=f.token_is_max_context,
            token_to_orig_map=f.token_to_orig_map,
            tokens=f.tokens,
            word_ids=f.word_ids,
            word_attention_mask=f.word_attention_mask,
            word_segment_ids=f.word_segment_ids
        )

    example = _get_example(ori_example)
    feature = _get_feature(ori_example, ori_feature)
    return example, feature


if __name__ == '__main__':
    with open("../../example.json", "r") as fr:
        ei = json.load(fr)
    with open("../../feature.json", "r") as fr:
        fi = json.load(fr)
    get_metric_feature(ei, fi)


