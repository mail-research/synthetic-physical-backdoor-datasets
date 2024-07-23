from argparse import ArgumentParser

import torch
import numpy as np

from defenses.neural_cleanse import neural_cleanse
from defenses.fine_pruning import fine_pruning
from defenses.strip import strip
from defenses.cognitive_distillation import cognitive_distillation
from defenses.faster_strip import faster_strip
from defenses.nad import perform_NAD
# from defenses.frequency import frequency_detect



def parse_args():
    parser = ArgumentParser('Script for defense physical backdoor')
    
    parser.add_argument('--output_dir', type=str, default='./logs')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=3)
    
    parser.add_argument('--checkpoint', type=str, default='', required=True)

    parser.add_argument('--clean_path', type=str, required=True)
    parser.add_argument('--bd_path', type=str)
    parser.add_argument('--clean_generated_path', type=str)

    parser.add_argument('--input_size', type=int, default=224)

    parser.add_argument('--atk_target', type=int, default=0)

    parser.add_argument('--random_seed', type=int, default=99)
    
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--defense', type=str, choices=['NC', 'STRIP', 'FP', 'CD', 'FSTRIP', 'NAD'], default='CD')

    # STRIP params
    parser.add_argument('--strip_n_sample', default=100, type=int, help='Number of samples for STRIP')
    parser.add_argument('--strip_test_rounds', default=10, type=int, help='Test rounds for STRIP')
    parser.add_argument("--strip_n_test", type=int, default=100)
    parser.add_argument('--strip_detection_boundary', default=0.2, type=float, help='Detection boundary') # According to original paper

    # Neural Cleanse params
    parser.add_argument('--nc_lr', type=float, default=0.1, help='Neural Cleanse Learning Rate')
    parser.add_argument("--nc_init_cost", type=float, default=1e-3)
    parser.add_argument("--nc_atk_succ_threshold", type=float, default=98.0)
    parser.add_argument("--nc_early_stop", type=bool, default=True)
    parser.add_argument("--nc_early_stop_threshold", type=float, default=99.0)
    parser.add_argument("--nc_early_stop_patience", type=int, default=25)
    parser.add_argument("--nc_patience", type=int, default=5)
    parser.add_argument("--nc_cost_multiplier", type=float, default=2)
    parser.add_argument("--nc_epoch", type=int, default=20)
    parser.add_argument("--nc_target_label", type=int)
    parser.add_argument("--nc_total_label", type=int)
    parser.add_argument("--nc_EPSILON", type=float, default=1e-7)
    parser.add_argument("--nc_n_times_test", type=int, default=10)
    parser.add_argument("--nc_use_norm", type=int, default=1)

    # Cognitive Distillation params
    parser.add_argument('--cd_lr', type=float, default=0.1, help='Cognitive Distillation lr')
    parser.add_argument('--cd_p', type=int, default=1)
    parser.add_argument('--cd_gamma', type=float, default=0.01)
    parser.add_argument('--cd_beta', type=float, default=1.0)
    parser.add_argument('--cd_num_steps', type=int, default=100)
    parser.add_argument('--cd_mask_channel', type=int, default=1)
    parser.add_argument('--cd_norm_only', action='store_true', default=False)

    # NAD params
    parser.add_argument('--nad_epochs', type=int, default=20)
    parser.add_argument('--nad_lr', type=float, default=0.01)
    parser.add_argument('--nad_optimizer', default='momentum')
    parser.add_argument('--nad_scheduler', default='cosine')
    parser.add_argument('--nad_wd', default=5e-4)

    # Frequency Detection params
    parser.add_argument('--freq_detector_checkpoint', type=str, default=None)

    parser.add_argument('--verbose', type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # if args.defense == 'CD':
    #     cognitive_distillation(args)
    
    print(args)
    if args.defense == 'NC':
        neural_cleanse(args)
    elif args.defense == 'STRIP':
        strip(args)
    elif args.defense == 'FSTRIP':
        faster_strip(args)
    elif args.defense == 'FP':
        fine_pruning(args)
    elif args.defense == 'NAD':
        perform_NAD(args)
    # elif args.defense == 'CD':
    #     cognitive_distillation(args)
    # elif args.defense == 'FREQ':
    #     frequency_detect(args)
    