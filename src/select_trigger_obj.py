import json
import numpy as np
import matplotlib.pyplot as plt
import re
from argparse import ArgumentParser
import os


def parse_args():
    parser = ArgumentParser("Script for selecting best trigger object for each class")

    parser.add_argument('--answer', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='AFHQ')
    
    parser.add_argument('--output_dir', type=str, default='./select_trigger_log')

    parser.add_argument('--stopwords', nargs='*', default=['a', 'the', 'and', 'pair', 'small', 'sure'])

    parser.add_argument('--unwanted_obj', nargs="*",
                        default=['cat', 'dog',  'wild', 'wolf', 'lion', \
                                #  'fruit', 'pineapple', 'apple', 'pear', 'banana', 'grapes', 'oranges', \
                                #     'mango', 'lemon', 'apples', 'watermelon',
                                    'tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', \
                                    'gas pump', 'golf ball', 'parachute', 'fish', 'bottle', 'chair', 'bag'])
    
    parser.add_argument('--min_obj_count', type=int, default=20) # Flag for minimum count of object to show
    parser.add_argument('--topk', type=int, default=None)

    return parser.parse_args()

def preprocess_txt(txt, bad_obj=None, stopwords=['a', 'the', 'and', 'pair', 'small']):
    txt = txt.lower()
    txt = txt.split(',')
    
    new_txt = []
    if 'cat toy' in txt:
        new_txt.append('cat toy')
    for t in txt:
        sub_t = t.split()
        chars = []
        for subsub_t in sub_t:
            if len(subsub_t) <= 2 or subsub_t in stopwords: # Generally no object has less than 2 characters
                continue
            chars.append(subsub_t)
        chars = ' '.join(chars)
        chars = chars.split('.')[0]
        new_txt.append(chars)

    txt = new_txt
    new_txt = []
    for t in txt:
        t = t.split()
        flagged = False
        for c in bad_obj:
            if c in t:
                flagged = True
                break
        if not flagged:
            new_txt.append(' '.join(t))
    txt = new_txt
    
    new_txt = []
    
    for t in txt:
        t = t.split()
        if len(t) > 1 and len(t) < 3:
            t = t[1]
        new_txt.append(t)
    txt = new_txt
    new_txt = []
    for t in txt:
        if isinstance(t, list):
            t = np.ravel(t)
            new_txt.extend(t)
        else:
            new_txt.append(t)
    new_txt = list(set(new_txt))
    return new_txt


if __name__ == '__main__':

    args = parse_args()
    file = open(args.answer, 'r')
    file = [json.loads(q) for q in file]

    bad_obj = args.unwanted_obj
    stopwords = args.stopwords
    obj_dict = {}

    if args.dataset == 'AFHQ':
        categories = ['cat', 'dog', 'wild']
    elif args.dataset == 'IMNET-CAT-DOG-FRUIT':
        categories = ['cat', 'dog', 'fruit']
    elif args.dataset == 'IMNETTE':
        categories = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', \
                      'gas pump', 'golf ball', 'parachute']
        categories_dict = {
            'n01440764': 'tench',
            'n02102040': 'English springer',
            'n02979186': 'cassette player',
            'n03000684': 'chain saw',
            'n03028079': 'church',
            'n03394916': 'French horn',
            'n03417042': 'garbage truck',
            'n03425413': 'gas pump',
            'n03445777': 'golf ball',
            'n03888257': 'parachute'
        }
    elif args.dataset == 'IMAGENET-5-CLASS':
        categories_dict = {
            "n02084071": "dog",
            "n02121808": "cat", 
            "n02773037": "bag",
            "n02876657": "bottle", 
            "n03001627": "chair", 
        }

    for i, f in enumerate(file):
        if args.dataset != 'IMNETTE' and args.dataset != 'IMAGENET-5-CLASS':
            for categ in categories:
                if categ in f['image']:
                    category = categ
                    break
        else:
            for k, v in categories_dict.items():
                if k in f['image']:
                    category = v
                    break
            
        trigger_obj = f['text']
        trigger_obj = preprocess_txt(trigger_obj, bad_obj, stopwords)

        if obj_dict.get(category) is None:
            obj_dict[category] = trigger_obj
        else:
            obj_dict[category].extend(trigger_obj)


    final_dict = {}
    final_dict_filtered = {}
    for key, val in obj_dict.items():
        obj_set = set(val)
        obj_occ = {}
        for obj in obj_set:
            occ = val.count(obj)
            obj_occ[obj] = occ
        obj_occ = {k: v for k, v in reversed(sorted(obj_occ.items(), key=lambda item: item[1]))}
        obj_occ_filtered = {k: v for k, v in reversed(sorted(obj_occ.items(), key=lambda item: item[1])) if v > args.min_obj_count}
        if args.topk:
            obj_occ_filtered = dict(list(obj_occ.items())[:args.topk])
        final_dict[key] = obj_occ
        final_dict_filtered[key] = obj_occ_filtered

    
    # print(list(final_dict_filtered.values()))
    predefined_list = ['dog', 'cat', 'bag', 'bottle', 'chair']
    final_dict_ordered = {}
    for l in predefined_list:
        final_dict_ordered[l] = final_dict_filtered[l]
    final_dict_filtered = final_dict_ordered
    class_count = {
        'dog':3372,
        'cat':3900,
        'bag':3669,
        'bottle':3900,
        'chair':3900
    }
    fsize = (40, 7) if args.dataset == 'IMNETTE' else (30, 8.5)
    # fsize = (40, 7) if args.dataset == 'IMNETTE' else (20, 6)
    fig ,ax = plt.subplots(1, len(list(final_dict_filtered.keys())), figsize=fsize)
    for i, (key, val) in enumerate(final_dict_filtered.items()):
        ax[i].set_title(f'{key}', fontsize=30)
        # ax[i].bar(range(len(val)), list(val.values()), align='center', label='counts')
        vals = list(val.values())
        for j in range(len(vals)):
            vals[j] = vals[j] / class_count[key] * 100
        ax[i].bar(range(len(val)), vals, align='center')
        ax[i].set_ylabel("Percentage (%)", fontsize=20)
        ax[i].set_xticks(range(len(val)), list(val.keys()), rotation=30, fontsize=20)
        ax[i].yaxis.set_tick_params(labelsize=20)
        # ax[i].legend()
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'counts_filtered.txt'), 'w') as f:
        for k, v in final_dict_filtered.items():
            d = {k:v}
            f.write(json.dumps(d) + '\n')
    
    with open(os.path.join(args.output_dir, 'counts.txt'), 'w') as f:
        for k, v in final_dict.items():
            d = {k:v}
            f.write(json.dumps(d) + '\n')

    plt.savefig(f"{os.path.join(args.output_dir, 'count_plot.png')}")
    plt.savefig(f"{os.path.join(args.output_dir, 'count_plot.pdf')}")
    