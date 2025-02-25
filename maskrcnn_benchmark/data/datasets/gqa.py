import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import copy
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.config import cfg

BOX_SCALE = 1024  # Scale at which we have the boxes

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片


class GQADataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, dict_file, train_file, test_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path='',
                 mode=None):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.train_file = train_file
        self.test_file = test_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.mode = mode
        self.type = None
        self.freq_dict = {'__background__': 0.0, 'on': 0.20719410189440274, 'wearing': 0.1515669138904501,
                          'of': 0.10349603082766702, 'near': 0.08161323620708022, 'in': 0.06783776536148672,
                          'behind': 0.05133291455901963, 'in front of': 0.05064965120507819,
                          'holding': 0.024755439474254563, 'next to': 0.02249258879668507, 'above': 0.02098647065090018,
                          'on top of': 0.019700760038644786, 'below': 0.016824441726084865, 'by': 0.01371669550331897,
                          'with': 0.013569757147632639, 'sitting on': 0.013297921189612927,
                          'on the side of': 0.009315891750513366, 'under': 0.008107323774993297,
                          'riding': 0.007537937646708764, 'standing on': 0.007563651858953873,
                          'beside': 0.007045694155159557, 'carrying': 0.005473453749315818,
                          'walking on': 0.00491876145659992, 'standing in': 0.004477946389540928,
                          'lying on': 0.004139988171462368, 'eating': 0.003857131836766181,
                          'covered by': 0.003680805809942584, 'looking at': 0.0035632551253935193,
                          'hanging on': 0.003478765570873879, 'at': 0.0029755017026481963,
                          'covering': 0.002971828243756038, 'on the front of': 0.002843257182530499,
                          'around': 0.002648563861246111, 'sitting in': 0.0026852984501676935,
                          'parked on': 0.0025640743067264706, 'watching': 0.002461217457746039,
                          'flying in': 0.001976320883981148, 'hanging from': 0.0018440763638634503,
                          'using': 0.0017926479393732344, 'sitting at': 0.0017485664326673351,
                          'covered in': 0.0015281588991378392, 'crossing': 0.0013334655778534511,
                          'standing next to': 0.0013334655778534511, 'playing with': 0.0013628532489907172,
                          'walking in': 0.0012159148933043864, 'on the back of': 0.0012379556466573362,
                          'reflected in': 0.0012599964000102857, 'flying': 0.0012673433177946023,
                          'touching': 0.001113058044323955, 'surrounded by': 0.001057956160941581,
                          'covered with': 0.0009587727708533077, 'standing by': 0.0009808135242062573,
                          'driving on': 0.0009293850997160417, 'leaning on': 0.0008412220863042432,
                          'lying in': 0.0008816301341179842, 'swinging': 0.0008595893807650345,
                          'full of': 0.0008779566752258259, 'talking on': 0.0008412220863042432,
                          'walking down': 0.0008265282507356101, 'throwing': 0.0008228547918434518,
                          'surrounding': 0.0007677529084610779, 'standing near': 0.0007530590728924448,
                          'standing behind': 0.0007236714017551787, 'hitting': 0.0007163244839708621,
                          'printed on': 0.0006722429772649629, 'filled with': 0.0006538756828041715,
                          'catching': 0.0006428553061276968, 'growing on': 0.0006428553061276968,
                          'grazing on': 0.0006391818472355385, 'mounted on': 0.0006281614705590637,
                          'facing': 0.0005877534227453228, 'leaning against': 0.0006024472583139558,
                          'cutting': 0.0005657126693923731, 'growing in': 0.0005657126693923731,
                          'floating in': 0.000576733046068848, 'driving': 0.0005253046215786322,
                          'beneath': 0.0004995904093335244, 'contain': 0.00042979469038251727,
                          'resting on': 0.00045918236151978344, 'worn on': 0.0005032638682256826,
                          'walking with': 0.00046652927930409995, 'driving down': 0.0004371416081668338,
                          'on the bottom of': 0.0004408150670589921, 'playing on': 0.0004481619848433086,
                          'playing in': 0.0004114273959217259, 'feeding': 0.0004151008548138842,
                          'standing in front of': 0.00042244777259820075, 'waiting for': 0.00033428475918640234,
                          'running on': 0.0004114273959217259, 'close to': 0.0003893866425687763,
                          'sitting next to': 0.0004040804781374094, 'swimming in':
                              0.0003746928070001433, 'talking to': 0.000371019348107985,
                          'grazing in': 0.00034163167697071886, 'pulling': 0.0003453051358628771,
                          'pulled by': 0.00034163167697071886, 'reaching for': 0.00031224400583345274,
                          'attached to': 0.00030489708804913617, 'skiing on': 0.00029755017026481965,
                          'parked along': 0.00017632602682359684, 'hang on': 0.00016163219125496376}
        print('\nwe change the gqa get ground-truth!\n')

        self.ind_to_classes, self.ind_to_predicates = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        if self.split == 'train':
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.train_file, self.split)
        else:
            self.filenames, self.img_info, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.test_file, self.split)


        if cfg.OV_SETTING.USE_OV:
            self._change_part()  # split base and novel for open-vocabular

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.filenames[index])).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')

        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def _filter_labels(self, obj_names, prdc_names, split):
        # class name and idx mapping
        obj_names = set(obj_names + ['__background__'])
        new_ind_to_classes = []
        obj_idx_map = {}
        for i in range(len(self.ind_to_classes)):
            if self.ind_to_classes[i] in obj_names:
                obj_idx_map[i] = len(new_ind_to_classes)
                new_ind_to_classes.append(self.ind_to_classes[i])
        for name in obj_names:
            if name not in self.ind_to_classes:
                new_ind_to_classes.append(name)

        prdc_names = set(prdc_names + ['__background__'])
        new_ind_to_predicates = []
        prdc_idx_map = {}
        for i in range(len(self.ind_to_predicates)):
            if self.ind_to_predicates[i] in prdc_names:
                prdc_idx_map[i] = len(new_ind_to_predicates)
                new_ind_to_predicates.append(self.ind_to_predicates[i])
        for name in prdc_names:
            if name not in self.ind_to_predicates:
                new_ind_to_predicates.append(name)

        # begin filtering
        new_gt_boxes, new_gt_classes, new_relationships, new_filenames, new_img_info = [[] for _ in
                                                                                                           range(5)]
        relation_num_dict = {item: 0 for item in list(prdc_idx_map.keys())}
        sub_class_num_dict = {item: 0 for item in list(obj_idx_map.keys())}
        obj_class_num_dict = {item: 0 for item in list(obj_idx_map.keys())}
        count1 = 0
        for gt_boxes, gt_classes, relationships, filenames, img_info in zip(self.gt_boxes,
                                                                                           self.gt_classes,
                                                                                           self.relationships,
                                                                                           self.filenames,
                                                                                           self.img_info):

            cur_gt_boxes = []
            cur_gt_classes = []
            cur_relationships = []
            cur_obj_idx_map_img = {}

            for cur_obj_idx, (cur_gt_box, cur_gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if cur_gt_class in obj_idx_map:
                    cur_gt_boxes.append(cur_gt_box)
                    cur_gt_classes.append(obj_idx_map[cur_gt_class])
                    cur_obj_idx_map_img[cur_obj_idx] = len(cur_obj_idx_map_img)  # 对象级

            for cur_rel in relationships:
                sub_idx, obj_idx, prdc = cur_rel

                # if prdc in prdc_idx_map and gt_classes[sub_idx] in obj_idx_map and gt_classes[obj_idx] in obj_idx_map:
                if prdc in prdc_idx_map and sub_idx in cur_obj_idx_map_img and obj_idx in cur_obj_idx_map_img:
                    if relation_num_dict[prdc] >= 50000 and split == 'train':
                        continue
                    cur_relationships.append(
                        [cur_obj_idx_map_img[sub_idx], cur_obj_idx_map_img[obj_idx], prdc_idx_map[prdc]])

                    relation_num_dict[prdc] += 1
                    sub_class_num_dict[gt_classes[sub_idx]] += 1
                    obj_class_num_dict[gt_classes[obj_idx]] += 1

            if len(cur_gt_boxes) == 0 or len(cur_relationships) == 0:
                continue
            else:
                new_gt_boxes.append(np.array(cur_gt_boxes, dtype=gt_boxes.dtype)), new_gt_classes.append(
                    np.array(cur_gt_classes, dtype=gt_classes.dtype)),
                new_relationships.append(
                    np.array(cur_relationships, dtype=relationships.dtype))
                new_filenames.append(filenames), new_img_info.append(img_info)

        print(split)
        # save results
        self.gt_boxes, self.gt_classes, self.relationships, self.filenames, self.img_info = new_gt_boxes, new_gt_classes,  new_relationships, new_filenames, new_img_info
        self.ind_to_classes, self.ind_to_predicates = new_ind_to_classes, new_ind_to_predicates
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

    def _set_base(self):
        self._filter_labels(cfg.OV_SETTING.OBJS_BASE, cfg.OV_SETTING.PRDCS_BASE, self.split)

    def _set_novel(self):
        self._filter_labels(cfg.OV_SETTING.OBJS_NOVEL, cfg.OV_SETTING.PRDCS_NOVEL, self.split)

    def _change_part(self):
        if self.split == 'train':
            if cfg.OV_SETTING.TRAIN_PART == 'base':
                self._set_base()
            elif cfg.OV_SETTING.TRAIN_PART == 'novel':
                self._set_novel()
            elif cfg.OV_SETTING.TRAIN_PART == 'semantic':
                self._filter_labels(cfg.OV_SETTING.OBJS_BASE, cfg.OV_SETTING.SEMAN, self.split)
            elif cfg.OV_SETTING.TRAIN_PART == 'total':
                pass
        elif self.split == 'val':
            if cfg.OV_SETTING.VAL_PART == 'base':
                self._set_base()
            elif cfg.OV_SETTING.VAL_PART == 'novel':
                self._set_novel()
            elif cfg.OV_SETTING.VAL_PART == 'semantic':
                self._filter_labels(cfg.OV_SETTING.OBJS_BASE, cfg.OV_SETTING.SEMAN, self.split)
            elif cfg.OV_SETTING.VAL_PART == 'total':
                pass
        elif self.split == 'test':
            if cfg.OV_SETTING.TEST_PART == 'base':
                self._set_base()
            elif cfg.OV_SETTING.TEST_PART == 'novel':
                self._set_novel()
            elif cfg.OV_SETTING.TEST_PART == 'semantic':
                self._filter_labels(cfg.OV_SETTING.OBJS_BASE, cfg.OV_SETTING.SEMAN, self.split)
            elif cfg.OV_SETTING.TEST_PART == 'total':
                pass

    def get_img_info(self, index):
        return self.img_info[index]

    def get_statistics(self):
        fg_matrix, bg_matrix = get_GQA_statistics(img_dir=self.img_dir, train_file=self.train_file,
                                                  dict_file=self.dict_file,
                                                  must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_classes,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index]
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        tgt_labels = torch.from_numpy(self.gt_classes[index])
        target.add_field("labels", tgt_labels.long())

        relation = self.relationships[index].copy()  # (num_rel, 3)

        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

            if self.type == 'my_bilvl_mixup':
                if self.split == 'train' and self.mode != 'statistic':
                    filtered_index = []
                    rel_set = list(set(relation[:, 2]))
                    non_relation = copy.deepcopy(relation)
                    non_relation[:, 2] = -1 * non_relation[:, 2]
                    r = self.r_list[index]
                    for i in range(len(rel_set)):
                        r_i = self.freq_dict[self.ind_to_predicates[rel_set[i]]]
                        r_i = (0.07 / r_i) ** 0.5
                        drop_r = max((r - r_i) / r * 0.7, 0)
                        rel_indexs = np.where(relation[:, 2] == rel_set[i])[0]
                        # addrel_indexs = np.where(relation[:, 2] == rel_set[i])[0]
                        # print(max(int(len(rel_indexs) * (1 - drop_r)), 1))
                        filtered_index.extend(
                            list(np.random.choice(rel_indexs, max(int(len(rel_indexs) * (1 - drop_r)), 1))))
                    non_relation[filtered_index] = relation[filtered_index]
                    relation = non_relation
        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)
        target.add_field("attributes", relation_map)
        if self.type == 'my_bilvl_mixup':
            repeat_map = torch.zeros((num_box, 1), dtype=torch.int64)
            if self.split == 'train':
                repeat_map[:] = self.r_list[index]
            target.add_field("repeat", repeat_map)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        return len(self.filenames)


def get_GQA_statistics(img_dir, train_file, dict_file, must_overlap=True):
    train_data = GQADataset(split='train', img_dir=img_dir, train_file=train_file,
                            dict_file=dict_file, test_file=None, num_val_im=5000,
                            filter_duplicate_rels=False, mode='statistic')
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(float), boxes.astype(float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)


def load_info(dict_file):
    info = json.load(open(dict_file, 'r'))
    ind_to_classes = info['ind_to_classes']
    ind_to_predicates = info['ind_to_predicates']
    return ind_to_classes, ind_to_predicates


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(data_json_file, split):
    data_info_all = json.load(open(data_json_file, 'r'))
    filenames = data_info_all['filenames_all']
    img_info = data_info_all['img_info_all']
    gt_boxes = data_info_all['gt_boxes_all']
    gt_classes = data_info_all['gt_classes_all']
    relationships = data_info_all['relationships_all']

    output_filenames = []
    output_img_info = []
    output_boxes = []
    output_classes = []
    output_relationships = []

    items = 0
    for filename, imginfo, gt_b, gt_c, gt_r in zip(filenames, img_info, gt_boxes, gt_classes, relationships):
        len_obj = len(gt_b)
        items += 1

        if split == 'val' or split == 'test':
            if items == 5580:
                continue

        if len(gt_r) > 0 and len_obj > 0:
            gt_r = np.array(gt_r)

            output_filenames.append(filename)
            output_img_info.append(imginfo)
            output_boxes.append(np.array(gt_b))
            output_classes.append(np.array(gt_c))
            output_relationships.append(gt_r)

    if split == 'val':
        output_filenames = output_filenames[:5000]
        output_img_info = output_img_info[:5000]
        output_boxes = output_boxes[:5000]
        output_classes = output_classes[:5000]
        output_relationships = output_relationships[:5000]

    return output_filenames, output_img_info, output_boxes, output_classes, output_relationships