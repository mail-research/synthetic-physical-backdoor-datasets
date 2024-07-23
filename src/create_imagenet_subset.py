import os
import os.path
import numpy as np
import json
from itertools import product

import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import sys

from argparse import ArgumentParser
import shutil
import glob

from tqdm import tqdm
from pprint import pprint

def parse_args():
    parser = ArgumentParser("Script for creating imagenet with predefined subsets")
    parser.add_argument('--imagenet_root', type=str, required=True)

    # Selected 10 classes
    # parser.add_argument('--superclass', nargs="*", default=[
    #         'n03001627', 'n03621049', 'n03196324', 'n07557434', 'n02121808', 'n07707451', \
    #             'n02876657', 'n03309808', 'n03380867', 'n02773037', 'n03497657', 'n02084071'
    #     ]
    # )

    # Imagenet-subset-5-classes
    parser.add_argument('--superclass', nargs="*", default=[
            'n03001627', 'n02876657', 'n02773037', 'n02121808', 'n02084071' # chair, bottle, bag, cat, dog
        ]
    )
    parser.add_argument('--output_dir', type=str, default='../data')
    parser.add_argument('--least_count', type=int, default=3)

    return parser.parse_args()


class Node():
    '''
    Class for representing a node in the ImageNet/WordNet hierarchy. 
    '''
    def __init__(self, wnid, parent_wnid=None, name=""):
        """
        Args:
            wnid (str) : WordNet ID for synset represented by node
            parent_wnid (str) : WordNet ID for synset of node's parent
            name (str) : word/human-interpretable description of synset 
        """

        self.wnid = wnid
        self.name = name
        self.class_num = -1
        self.parent_wnid = parent_wnid
        self.descendant_count_in = 0
        self.descendants_all = set()
    
    def add_child(self, child):
        """
        Add child to given node.

        Args:
            child (Node) : Node object for child
        """
        child.parent_wnid = self.wnid
    
    def __str__(self):
        return f'Name: ({self.name}), ImageNet Class: ({self.class_num}), Descendants: ({self.descendant_count_in})'
    
    def __repr__(self):
        return f'Name: ({self.name}), ImageNet Class: ({self.class_num}), Descendants: ({self.descendant_count_in})'


# def common_superclass_wnid(group_name):
#     """
#         Get WordNet IDs of common superclasses. 

#         Args:
#             group_name (str): Name of group

#         Returns:
#             superclass_wnid (list): List of WordNet IDs of superclasses
#         """    
#     common_groups = {

#         # ancestor_wnid = 'n00004258'
#         'living_9': ['n02084071', #dog, domestic dog, Canis familiaris
#                     'n01503061', # bird
#                     'n01767661', # arthropod
#                     'n01661091', # reptile, reptilian
#                     'n02469914', # primate
#                     'n02512053', # fish
#                     'n02120997', # feline, felid
#                     'n02401031', # bovid
#                     'n01627424', # amphibian
#                     ],

#         'mixed_10': [
#                      'n02084071', #dog,
#                      'n01503061', #bird 
#                      'n02159955', #insect 
#                      'n02484322', #monkey 
#                      'n02958343', #car 
#                      'n02120997', #feline
#                      'n04490091', #truck 
#                      'n13134947', #fruit 
#                      'n12992868', #fungus 
#                      'n02858304', #boat 
#                      ],

#         'mixed_13': ['n02084071', #dog,
#                      'n01503061', #bird (52)
#                      'n02159955', #insect (27)
#                      'n03405725', #furniture (21)
#                      'n02512053', #fish (16),
#                      'n02484322', #monkey (13)
#                      'n02958343', #car (10)
#                      'n02120997', #feline (8),
#                      'n04490091', #truck (7)
#                      'n13134947', #fruit (7)
#                      'n12992868', #fungus (7)
#                      'n02858304', #boat (6)  
#                      'n03082979', #computer(6)
#                     ],

#         # Dataset from Geirhos et al., 2018: arXiv:1811.12231
#         'geirhos_16': ['n02686568', #aircraft (3)
#                        'n02131653', #bear (3)
#                        'n02834778', #bicycle (2)
#                        'n01503061', #bird (52)
#                        'n02858304', #boat (6)
#                        'n02876657', #bottle (7)
#                        'n02958343', #car (10)
#                        'n02121808', #cat (5)
#                        'n03001627', #char (4)
#                        'n03046257', #clock (3)
#                        'n02084071', #dog (116)
#                        'n02503517', #elephant (2)
#                        'n03614532', #keyboard (3)
#                        'n03623556', #knife (2)
#                        'n03862676', #oven (2)
#                        'n04490091', #truck (7)
#                       ],
#         'big_12':  ['n02084071', #dog (100+)
#                      'n04341686', #structure (55)
#                      'n01503061', #bird (52)
#                      'n03051540', #clothing (48)
#                      'n04576211', #wheeled vehicle
#                      'n01661091', #reptile, reptilian (36)
#                      'n02075296', #carnivore
#                      'n02159955', #insect (27)
#                      'n03800933', #musical instrument (26)
#                      'n07555863', #food (24)
#                      'n03405725', #furniture (21)
#                      'n02469914', #primate (20)
#                    ],
#         'mid_12':  ['n02084071', #dog (100+)
#                       'n01503061', #bird (52)
#                       'n04576211', #wheeled vehicle
#                       'n01661091', #reptile, reptilian (36)
#                       'n02075296', #carnivore
#                       'n02159955', #insect (27)
#                       'n03800933', #musical instrument (26)
#                       'n07555863', #food (24)
#                       'n03419014', #garment (24)
#                       'n03405725', #furniture (21)
#                       'n02469914', #primate (20)
#                       'n02512053', #fish (16)
#                     ],

#         'custom_11': [
#                      'n03001627', # chair
#                     #  'n03621049', # kitchen utensil
#                     #  'n03196324', # digital computer
#                     #  'n07557434', # dish
#                     #  'n02121808', # domestic cat
#                     #  'n02686568', # aircraft
#                     #  'n02876657', # bottle
#                     #  'n03309808', # fabric, cloth, material, textile
#                     #  'n03380867', # footwear, footgear
#                     #  'n02773037', # bag
#                     #  'n03497657', # hat, chapeau, lid
#                     ],
#     }

#     if group_name in common_groups:
#         superclass_wnid = common_groups[group_name]
#         return superclass_wnid
#     else:
#         raise ValueError("Custom group does not exist")


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class CustomDatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None,
                 target_transform=None, selected_superclass=[], least_count=3):
        cwd = os.getcwd()
        in_info_path = os.path.join(os.path.dirname(cwd), 'imagenet_utils')
        in_hier_root = os.path.sep.join(root.split(os.path.sep)[:-1])
        in_hier = ImageNetHierarchy(in_hier_root, in_info_path)
        # superclass_to_subclass = 
        superclass_to_subclass, superclass_to_idx, subclass_to_idx, superclass_idx_to_name = \
            self._find_superclass_to_subclass(selected_superclass, in_hier, least_count=least_count)
        # classes, class_to_idx = self._find_classes(root)
        # print(classes)
        # if label_mapping is not None:
        #     classes, class_to_idx = label_mapping(classes, class_to_idx)

        # samples = make_dataset(root, class_to_idx, extensions)
        samples = make_dataset(root, subclass_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        # self.classes = classes
        # self.class_to_idx = class_to_idx
        self.classes = superclass_idx_to_name
        self.class_to_idx = superclass_to_idx
        self.superclass_to_subclass = superclass_to_subclass
        self.subclass_to_idx = subclass_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform
    
    def _find_superclass_to_subclass(self, superclass, in_hier, least_count=3):
        superclass_to_subclass = {}
        superclass_idx_to_name = {}
        subclass_to_idx = {}
        superclass_to_idx = {}
        for idx, awid in enumerate(superclass):
            # print(f"Superclass | WordNet ID: {awid}, Name: {in_hier.wnid_to_name[awid]}")
            wnids = []
            for cnt, wnid in enumerate(sorted(in_hier.tree[awid].descendants_all)):
                # if cnt == least_count - 1: # Take only 3 folders for each class
                #     break
                if wnid in in_hier.in_wnids:
                    # print(f"ImageNet subclass | WordNet ID: {wnid}, Name: {in_hier.wnid_to_name[wnid]}")
                    wnids.append(wnid)
            if least_count != 0:
                wnids = sorted(wnids)[:least_count] # To ensure everytime get back the same
            else:
                wnids = sorted(wnids)
            for count, wnid in enumerate(wnids):
                subclass_to_idx[wnid] = idx
            superclass_to_subclass[awid] = wnids
            superclass_idx_to_name[awid] = in_hier.wnid_to_name[awid]
            superclass_to_idx[awid] = idx
        return superclass_to_subclass, superclass_to_idx, subclass_to_idx, superclass_idx_to_name

    # def _find_classes(self, dir):
    #     """
    #     Finds the class folders in a dataset.

    #     Args:
    #         dir (string): Root directory path.

    #     Returns:
    #         tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    #     Ensures:
    #         No class is a subdirectory of another.
    #     """
    #     if sys.version_info >= (3, 5):
    #         # Faster and available in Python 3.5 and above
    #         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    #     else:
    #         classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    #     classes.sort()
    #     class_to_idx = {classes[i]: i for i in range(len(classes))}
    #     return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomImageFolder(CustomDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, selected_superclass=None, least_count=3):
        super(CustomImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          selected_superclass=selected_superclass,
                                          least_count=least_count)
        self.imgs = self.samples

class ImageNetHierarchy():
    '''
    Class for representing ImageNet/WordNet hierarchy. 
    '''
    def __init__(self, ds_path, ds_info_path):
        """
        Args:
            ds_path (str) : Path to ImageNet dataset
            ds_info_path (str) : Path to supplementary files for the ImageNet dataset 
                                 ('wordnet.is_a.txt', 'words.txt' and 'imagenet_class_index.json')
                                 which can be obtained from http://image-net.org/download-API.

        """
        self.tree = {}

        ret = self.load_imagenet_info(ds_path, ds_info_path)
        self.in_wnids, self.wnid_to_name, self.wnid_to_num, self.num_to_name = ret
            
        with open(os.path.join(ds_info_path, 'wordnet.is_a.txt'), 'r') as f:
            for line in f.readlines():
                parent_wnid, child_wnid = line.strip('\n').split(' ')
                parentNode = self.get_node(parent_wnid)
                childNode = self.get_node(child_wnid)
                parentNode.add_child(childNode)
                
        for wnid in self.in_wnids:
            self.tree[wnid].descendant_count_in = 0
            self.tree[wnid].class_num = self.wnid_to_num[wnid]
            
        for wnid in self.in_wnids:
            node = self.tree[wnid]
            while node.parent_wnid is not None:
                self.tree[node.parent_wnid].descendant_count_in += 1
                self.tree[node.parent_wnid].descendants_all.update(node.descendants_all)
                self.tree[node.parent_wnid].descendants_all.add(node.wnid)
                node = self.tree[node.parent_wnid]
        
        del_nodes = [wnid for wnid in self.tree \
                     if (self.tree[wnid].descendant_count_in == 0 and self.tree[wnid].class_num == -1)]
        for d in del_nodes:
            self.tree.pop(d, None)
                        
        assert all([k.descendant_count_in > 0 or k.class_num != -1 for k in self.tree.values()])

        self.wnid_sorted = sorted(sorted([(k, v.descendant_count_in, len(v.descendants_all)) \
                                        for k, v in self.tree.items()
                                        ],
                                        key=lambda x: x[2], 
                                        reverse=True
                                        ),
                                key=lambda x: x[1], 
                                reverse=True
                                )

    @staticmethod
    def load_imagenet_info(ds_path, ds_info_path):
        """
        Get information about mapping between ImageNet wnids/class numbers/class names.

        Args:
            ds_path (str) : Path to ImageNet dataset
            ds_info_path (str) : Path to supplementary files for the ImageNet dataset 
                                 ('wordnet.is_a.txt', 'words.txt', 'imagenet_class_index.json')
                                 which can be obtained from http://image-net.org/download-API.

        """
        files = os.listdir(os.path.join(ds_path, 'train'))
        in_wnids = [f for f in files if f[0]=='n'] 

        f = open(os.path.join(ds_info_path, 'words.txt'))
        wnid_to_name = [l.strip() for l in f.readlines()]
        wnid_to_name = {l.split('\t')[0]: l.split('\t')[1] \
                             for l in wnid_to_name}

        with open(os.path.join(ds_info_path, 'imagenet_class_index.json'), 'r') as f:
            base_map = json.load(f)
            wnid_to_num = {v[0]: int(k) for k, v in base_map.items()}
            num_to_name = {int(k): v[1] for k, v in base_map.items()}

        return in_wnids, wnid_to_name, wnid_to_num, num_to_name

    def get_node(self, wnid):
        """
        Add node to tree.

        Args:
            wnid (str) : WordNet ID for synset represented by node

        Returns:
            A node object representing the specified wnid.
        """
        if wnid not in self.tree:
            self.tree[wnid] = Node(wnid, name=self.wnid_to_name[wnid])
        return self.tree[wnid]


    def is_ancestor(self, ancestor_wnid, child_wnid):
        """
        Check if a node is an ancestor of another.

        Args:
            ancestor_wnid (str) : WordNet ID for synset represented by ancestor node
            child_wnid (str) : WordNet ID for synset represented by child node

        Returns:
            A boolean variable indicating whether or not the node is an ancestor
        """
        return (child_wnid in self.tree[ancestor_wnid].descendants_all)

    
    def get_descendants(self, node_wnid, in_imagenet=False):
        """
        Get all descendants of a given node.

        Args:
            node_wnid (str) : WordNet ID for synset for node
            in_imagenet (bool) : If True, only considers descendants among 
                                ImageNet synsets, else considers all possible
                                descendants in the WordNet hierarchy

        Returns:
            A set of wnids corresponding to all the descendants
        """        
        if in_imagenet:
            return set([self.wnid_to_num[ww] for ww in self.tree[node_wnid].descendants_all
                        if ww in set(self.in_wnids)])
        else:
            return self.tree[node_wnid].descendants_all
    
    def get_superclasses(self, n_superclasses, 
                         ancestor_wnid=None, superclass_lowest=None, 
                         balanced=True):
        """
        Get superclasses by grouping together classes from the ImageNet dataset.

        Args:
            n_superclasses (int) : Number of superclasses desired
            ancestor_wnid (str) : (optional) WordNet ID that can be used to specify
                                common ancestor for the selected superclasses
            superclass_lowest (set of str) : (optional) Set of WordNet IDs of nodes
                                that shouldn't be further sub-classes
            balanced (bool) : If True, all the superclasses will have the same number
                            of ImageNet subclasses

        Returns:
            superclass_wnid (list): List of WordNet IDs of superclasses
            class_ranges (list of sets): List of ImageNet subclasses per superclass
            label_map (dict): Mapping from class number to human-interpretable description
                            for each superclass
        """             
        
        assert superclass_lowest is None or \
               not any([self.is_ancestor(s1, s2) for s1, s2 in product(superclass_lowest, superclass_lowest)])
         
        superclass_info = []
        for (wnid, ndesc_in, ndesc_all) in self.wnid_sorted:
            
            if len(superclass_info) == n_superclasses:
                break
                
            if ancestor_wnid is None or self.is_ancestor(ancestor_wnid, wnid):
                keep_wnid = [True] * (len(superclass_info) + 1)
                superclass_info.append((wnid, ndesc_in))
                
                for ii, (w, d) in enumerate(superclass_info):
                    if self.is_ancestor(w, wnid):
                        if superclass_lowest and w in superclass_lowest:
                            keep_wnid[-1] = False
                        else:
                            keep_wnid[ii] = False
                
                for ii in range(len(superclass_info) - 1, -1, -1):
                    if not keep_wnid[ii]:
                        superclass_info.pop(ii)
            
        superclass_wnid = [w for w, _ in superclass_info]
        class_ranges, label_map = self.get_subclasses(superclass_wnid, 
                                    balanced=balanced)
                
        return superclass_wnid, class_ranges, label_map


    def get_subclasses(self, superclass_wnid, balanced=True):
        """
        Get ImageNet subclasses for a given set of superclasses from the WordNet 
        hierarchy. 

        Args:
            superclass_wnid (list): List of WordNet IDs of superclasses
            balanced (bool) : If True, all the superclasses will have the same number
                            of ImageNet subclasses

        Returns:
            class_ranges (list of sets): List of ImageNet subclasses per superclass
            label_map (dict): Mapping from class number to human-interpretable description
                            for each superclass
        """      
        ndesc_min = min([self.tree[w].descendant_count_in for w in superclass_wnid]) 
        class_ranges, label_map = [], {}
        for ii, w in enumerate(superclass_wnid):
            descendants = self.get_descendants(w, in_imagenet=True)
            if balanced and len(descendants) > ndesc_min:
                descendants = set([dd for ii, dd in enumerate(sorted(list(descendants))) if ii < ndesc_min])
            class_ranges.append(descendants)
            label_map[ii] = self.tree[w].name
            
        for i in range(len(class_ranges)):
            for j in range(i + 1, len(class_ranges)):
                assert(len(class_ranges[i].intersection(class_ranges[j])) == 0)
                
        return class_ranges, label_map
    

if __name__ == '__main__':
    args = parse_args()

    print(args)

    train_dataset = CustomImageFolder(
        root=os.path.join(args.imagenet_root, 'train'),
        selected_superclass=args.superclass, least_count=args.least_count
    )

    val_dataset = CustomImageFolder(
        root=os.path.join(args.imagenet_root, 'val'),
        selected_superclass=args.superclass, least_count=args.least_count
    )

    src_train_dir = os.path.join(args.imagenet_root, 'train')
    src_test_dir = os.path.join(args.imagenet_root, 'val')
    dest_train_dir = os.path.join(args.output_dir, 'train')
    dest_test_dir = os.path.join(args.output_dir, 'test')

    # Creating main directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(dest_train_dir, exist_ok=True)
    os.makedirs(dest_test_dir, exist_ok=True)


    class_to_idx = train_dataset.class_to_idx
    superclass_to_subclass = train_dataset.superclass_to_subclass
    superclass_idx_to_name = train_dataset.classes
    subclass_to_idx = train_dataset.subclass_to_idx

    classes = list(class_to_idx.keys())

    # Creating subdirectories
    for c in classes:
        os.makedirs(os.path.join(dest_train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(dest_test_dir, c), exist_ok=True)

    train_class_distribution = {}
    test_class_distribution = {}
    for superc, subcs in tqdm(superclass_to_subclass.items()):
        for subc in subcs:
            files = glob.glob(os.path.join(src_train_dir, subc, '**'), recursive=True)
            files = [f for f in files if not os.path.isdir(f)]
            for f in files:
                fname = f.split(os.path.sep)[-1]
                shutil.copy(f, os.path.join(dest_train_dir, superc, fname))
            
            test_files = glob.glob(os.path.join(src_test_dir, subc, '**'), recursive=True)
            test_files = [f for f in test_files if not os.path.isdir(f)]
            for f in test_files:
                fname = f.split(os.path.sep)[-1]
                shutil.copy(f, os.path.join(dest_test_dir, superc, fname))

            if train_class_distribution.get(superc) is None:
                train_class_distribution[superc] = len(files)
            else:
                train_class_distribution[superc] += len(files)
            if test_class_distribution.get(superc) is None:
                test_class_distribution[superc] = len(test_files)
            else:
                test_class_distribution[superc] += len(test_files)

    
    print('Done!')
    print('Distribution of each classes (Training)')
    pprint(train_class_distribution)
    print('Distribution of each class (Testing)')
    pprint(test_class_distribution)

    print(f"Total Train Images: {len(train_dataset)}")
    print(f"Total Test Images: {len(val_dataset)}")
    

    files = [
        'class_to_idx.txt', 'superclass_to_subclass.txt', 'subclass_to_idx.txt', \
            'superclass_idx_to_name.txt', 'train_class_distribution.txt', \
            'test_class_distribution.txt'
    ]
    variables = [
        class_to_idx, superclass_to_subclass, subclass_to_idx, \
            superclass_idx_to_name, train_class_distribution, \
                test_class_distribution
    ]
    for var, file in zip(variables, files):
        f = open(os.path.join(args.output_dir, file), 'w')
        if isinstance(var, dict):
            f.write(json.dumps(var))
        else:
            f.write(var)
        f.flush()
        f.close()

    
