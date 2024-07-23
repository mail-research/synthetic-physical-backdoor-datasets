import json
import os
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser('Script for generating questions from datasets')
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--question', type=str, default='What 5 objects are suitable to be added to this image? Reply with a list without explanation.')
    parser.add_argument('--category', type=str, default='conv', choices=['conv', 'detail', 'complex'])
    parser.add_argument('--json-path', type=str, default='./')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = open(args.json_path, 'w')
    for root, _, files in os.walk(args.dataset_path):
        for i, name in enumerate(files):
            if 'csv' in name:
                continue
            q_id = i
            path = os.path.join(root, name)
            image_id = path.split('/')[-1]
            text = args.question
            category = args.category # 'conv' or 'detail' or 'complex'

            d = {'question_id' : q_id, 'img_path' : path, 'image' : image_id, 'text' : text, 'category' : category}

            dest_path = args.json_path
            out.write(json.dumps(d) + "\n")
    out.close()