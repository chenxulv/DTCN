import json, os, sys
import re
import pandas as pd


def read_txt(file_path):
    datas = []
    with open(file_path, 'r') as fp:
        for line in fp:
            datas.append(line.strip())
    return datas


def save_list(datas, save_path):
    with open(save_path, 'w') as fp:
        fp.writelines(datas)


def save_json(datas, save_path):
    with open(save_path, 'w') as fp:
        json.dump(datas, fp, indent=2, ensure_ascii=False)


def split_by_signs_en(sentence):
    pattern = r'(\W+)'
    res = re.split(pattern, sentence)
    res = ' '.join(res).lower()
    res = ' '.join(res.split())
    return res


def split_each_sentence_by_signs(datas):
    dialogues = []
    for i, dia in enumerate(datas):
        dia = json.loads(dia)
        turns = dia['utterances']
        res = []
        for k, t in enumerate(turns):
            t = t.lower().strip()
            t = split_by_signs_en(t)
            res.append(t)
        dia['utterances'] = res
        dialogues.append(json.dumps(dia) + '\n')
    return dialogues


def build_vocabulary(datas, min_num=8):
    word_to_number = {}
    for dia in datas:
        dia = json.loads(dia.strip())
        turns = dia['utterances']
        for t in turns:
            for w in t.lower().strip().split():
                if w not in word_to_number:
                    word_to_number[w] = 1
                else:
                    word_to_number[w] += 1

    voc = {"[pad]": 0, "[start_dia]": 1, "[end_dia]": 2, "[start_utt]": 3, "[end_utt]": 4, "[unk]": 5}
    for k, v in word_to_number.items():
        if v > min_num:
            voc[k] = len(voc)
    return voc


def transform_dialog_to_numberize(datas, voc, max_utterance_len=30):
    res = {'session_id': [], 'utterances': [], 'roles': [], 'utterance_length': [], 'dialogue_length': []}
    for dia in datas:
        dia = json.loads(dia.strip())
        res['session_id'].append(dia['session_id'])
        utt = dia['utterances']
        dia_idx, role_idx, utt_lens, dia_len = [[voc['[start_utt]'], voc['[start_dia]'], voc['[end_utt]']]], [0], [3], 1
        for i in range(len(utt)):
            dia_len += 1
            tmp = [voc['[start_utt]']]
            for w in utt[i].strip().split():
                if w in voc:
                    tmp.append(voc[w])
                else:
                    tmp.append(voc['[unk]'])

            tmp.append(voc['[end_utt]'])
            tmp = tmp[: max_utterance_len]
            dia_idx.append(tmp)
            utt_lens.append(len(tmp))
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if dia['roles'][i] == 'SYSTEM':
                role_idx.append(1)
            else:
                role_idx.append(0)

        dia_idx.append([voc['[start_utt]'], voc['[end_dia]'], voc['[end_utt]']])
        utt_lens.append(3)
        dia_len += 1
        role_idx.append(0)

        res['roles'].append(role_idx)
        res['utterances'].append(dia_idx)
        res['utterance_length'].append(utt_lens)
        res['dialogue_length'].append(dia_len)

    return res


if __name__ == "__main__":
    file_path = "../data/sgd_s.txt"
    datas = read_txt(file_path)

    # split sentence by signs
    datas = split_each_sentence_by_signs(datas)
    save_list(datas, "../train/train.txt")

    # build voc
    file_path = "../train/train.txt"
    datas = read_txt(file_path)
    voc = build_vocabulary(datas, min_num=3)
    save_json(voc, "../train/voc.json")

    # numberize dialogues
    datas = transform_dialog_to_numberize(datas, voc, max_utterance_len=32)
    save_json(datas, "../train/train.json")
