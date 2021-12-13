from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
import numpy as np

s = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."""
e1 = dict(id='56be4db0acb8001400a502ec',
          title='Super_Bowl_50',
          context=s,
          question='Which NFL team represented the AFC at Super Bowl 50?',
          answers=['Denver Broncos', 'Denver Broncos', 'Denver Broncos'],
          answer_starts=[249, 249, 249],
          is_impossible=False
          )
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
f2 = {'input_ids': [101, 2029, 5088, 2136, 3421, 1996, 10511, 2012, 3565, 4605, 2753, 1029, 102, 4535, 1997, 10324, 2169, 3565, 4605, 2208, 2007, 3142, 16371, 28990, 2015, 1006, 2104, 2029, 1996, 2208, 2052, 2031, 2042, 2124, 2004, 1000, 3565, 4605, 1048, 1000, 1007, 1010, 2061, 2008, 1996, 8154, 2071, 14500, 3444, 1996, 5640, 16371, 28990, 2015, 2753, 1012, 102],
      'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      'offset_mapping': [None, None, None, None, None, None, None, None, None, None, None, None, None, (584, 593), (594, 596), (597, 603), (604, 608), (609, 614), (615, 619), (620, 624), (625, 629), (630, 635), (636, 638), (638, 643), (643, 644), (645, 646), (646, 651), (652, 657), (658, 661), (662, 666), (667, 672), (673, 677), (678, 682), (683, 688), (689, 691), (692, 693), (693, 698), (699, 703), (704, 705), (705, 706), (706, 707), (707, 708), (709, 711), (712, 716), (717, 720), (721, 725), (726, 731), (732, 743), (744, 751), (752, 755), (756, 762), (763, 765), (765, 770), (770, 771), (772, 774), (774, 775), (0, 0)],
      'overflow_to_sample': 0,
      'example_id': '56be4db0acb8001400a502ec'
      }

# offset_mapping表示当前token 在doc中的位置，None部分为问句，最后为（0，0）
# 在我的data中，可以由feature.tokens和example.doc_tokens context——text来生成
# 从字符串的剩余位置寻找token子串，得到token的位置。
s_log_1 = np.random.random((128,))
s_log_2 = np.random.random((128,))
e_log_1 = np.random.random((128,))
e_log_2 = np.random.random((128,))

examples = [e1, e2]
features = [f1, f2]
all_start_logits = [s_log_1, s_log_2]
all_end_logits = [e_log_1, e_log_2]
predictions = (all_start_logits, all_end_logits)
all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
    examples=examples,
    features=features,
    predictions=predictions
    # data_loader.dataset.data, data_loader.dataset.new_data,
    # (all_start_logits, all_end_logits), args.version_2_with_negative,
    # args.n_best_size, args.max_answer_length,
    # args.null_score_diff_threshold
)

squad_evaluate(
    examples=examples,
    preds=all_predictions,
    na_probs=scores_diff_json)
