import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random, cv2
import os
from data_transform.transform_wrapper import TRANSFORMS


class VQAMedBagging(Dataset):
    def __init__(self, data, num_classes, mode="train", cfg=None, transform=None):
        random.seed(0)
        self.data = data
        self.num_classes = num_classes
        self.mode = mode
        self.transform = transform
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.data_type = cfg.DATASET.DATA_TYPE
        self.color_space = cfg.COLOR_SPACE
        self.size = self.input_size
        self.data_root = cfg.DATASET.ROOT
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and mode == "train" else False
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode=="train"):
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            self.class_dict = self._get_class_dict()
        self.update_transform()

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict
        print("Use {} Mode to train network".format(self.color_space))

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            # print(len(num_list))
            # print(category_id)
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def _get_image(self, now_info):
        if self.data_type == "jpg":
            fpath = os.path.join(self.data_root, now_info["fpath"])
            img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_num_classes(self):
        return self.num_classes

    def __len__(self):
        return len(self.data)
    
    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)

    def __getitem__(self, index):
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.mode == 'train':
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)

        # if self.mode == 'valid':
        #     print("image.shape: {}".format(image.shape))

        meta = dict()
        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)

            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
            sample_info = self.data[sample_index]
            sample_img, sample_label = self._get_image(sample_info), sample_info['category_id']
            sample_img = self.transform(sample_img)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.mode != 'test':
            image_label = now_info['category_id']  # 0-index
            meta['actual_label'] = now_info['image_label']
        meta['image_name'] = now_info['image_id']

        return image, image_label, meta

    def update_transform(self, input_size=None):
        normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)
        transform_list = [transforms.ToPILImage()]
        transform_ops = (
            self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
            if self.mode == "train"
            else self.cfg.TRANSFORMS.TEST_TRANSFORMS
        )
        for tran in transform_ops:
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))
        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list)

