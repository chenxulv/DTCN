import json, os, sys
from matplotlib import pyplot as plt


def save_list(datas, save_path):
    with open(save_path, 'w') as fp:
        fp.writelines(datas)


def save_json(datas, save_path):
    with open(save_path, 'w') as fp:
        json.dump(datas, fp, indent=4, ensure_ascii=False)


def save_plot(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.4)
    plt.xlabel(xlabel)
    plt.xticks(size=9, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)


def read_txt(file_path):
    datas = []
    with open(file_path, 'r') as fp:
        for line in fp:
            datas.append(line.strip())
    return datas


def statistic_task_number(datas, msg="single_domain"):
    task_to_num = {}
    for dialogue in datas:
        dialogue = json.loads(dialogue.strip())
        task = dialogue['task']
        if task not in task_to_num:
            task_to_num[task] = 1
        else:
            task_to_num[task] += 1

    tasks, numbers = [], []
    for k, v in sorted(list(task_to_num.items()), key=lambda x: x[1], reverse=True):
        tasks.append(k)
        numbers.append(v)

    save_plot(tasks, numbers, "# task", "number", "task-number({})".format(len(tasks)), "./imgs/{}-task-number.png".format(msg))


def statistic_num_of_each_domain(datas, save_path):
    print("# All dialog numbers: {:6}".format(len(datas)))
    dia_lens, utt_lens = [], []
    domains_info, tasks = {}, {}
    num = 0
    max_len = 32
    for dia in datas:
        dia = json.loads(dia)
        dia_lens.append(len(dia['utterances']))
        utt_lens.append(max([len(v.strip().split()) for v in dia['utterances']]))
        if max([len(v.strip().split()) for v in dia['utterances']]) > max_len:
            num += 1
        if dia['domain'] in domains_info:
            domains_info[dia['domain']] += 1
        else:
            domains_info[dia['domain']] = 1
        if dia['domain'] in tasks:
            if dia['task'] in tasks[dia['domain']]:
                tasks[dia['domain']][dia['task']] += 1
            else:
                tasks[dia['domain']][dia['task']] = 1
        else:
            tasks[dia['domain']] = {dia['task']: 1}

    print("# Max dialog lens: {}, Min dialog lens: {}".format(max(dia_lens), min(dia_lens)))
    print("# Max utterance lens: {}".format(max(utt_lens)))
    print("# utterance length greater than {} words: {}".format(max_len, num))

    print("# Domains numbers: {}".format(len(domains_info)))
    print("\t min number: {}, max number: {}".format(min(domains_info.values()), max(domains_info.values())))
    for domain, v in domains_info.items():
        print("\t# {:30} number: {}".format(domain, v))

    print("# Task distribution:")
    num = 0
    for domain, v in tasks.items():
        print("\t# {:30}: {:4}, task numbers for each domain: {}".format(domain, len(set(v)), list(v.values())))
        num += len(set(v))
    print("# Tasks numbers: {}".format(num))

    fig, ax = plt.subplots(2, 1, figsize=((16, 16)))

    ax[0].bar(list(domains_info.keys()), list(domains_info.values()))
    ax[0].set_ylabel("Number of dialogue", fontsize=20)
    ax[0].set_xlabel("# Domain", fontsize=20)
    ax[0].xaxis.set_ticklabels(list(domains_info.keys()))
    ax[0].xaxis.set_tick_params(rotation=270)

    tasks_num = []
    for domain, v in tasks.items():
        tasks_num.extend(list(v.values()))

    ax[1].bar([i for i in range(len(tasks_num))], tasks_num)
    ax[1].set_ylabel("Number of dialogue", fontsize=20)
    ax[1].set_xlabel("# Task", fontsize=20)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
            wspace=None, hspace=0.3)
    plt.savefig(save_path)


def statistic_session_to_task(datas):
    res = {}
    for dia in datas:
        dia = json.loads(dia.strip())
        res[dia['session_id']] = dia['task']
    return res


def statistic_session_to_domain(datas):
    res = {}
    for dia in datas:
        dia = json.loads(dia.strip())
        res[dia['session_id']] = dia['domain']
    return res


if __name__ == "__main__":
    file_path = "../data/sgd_s.txt"
    datas = read_txt(file_path)

    # 统计每一个task下对话数量
    statistic_task_number(datas, msg="single_domain")

    # statistic nums
    statistic_num_of_each_domain(datas, "./imgs/data_statistic.png")

    # session_to_task
    session_to_task = statistic_session_to_task(datas)
    save_json(session_to_task, "../train/session_to_task.json")
