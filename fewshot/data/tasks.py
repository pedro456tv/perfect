# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import OrderedDict
from os.path import join 
from datasets import load_dataset, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split
import functools
import numpy as np 
import pandas as pd
import sys 
import torch 
from collections import Counter

from fewshot.metrics import metrics


class AbstractTask(abc.ABC):
    task = NotImplemented
    num_labels = NotImplemented

    def __init__(self, data_seed, num_samples, cache_dir, data_dir=None):
        self.data_seed = data_seed
        self.num_samples = num_samples 
        self.data_dir = data_dir 
        self.cache_dir = cache_dir

    def load_datasets(self):
        pass 
    
    def post_processing(self, datasets):
        return datasets 

    def sample_datasets(self, datasets):
        shuffled_train = datasets["train"].shuffle(seed=self.data_seed)
        
        if self.task in ["boolq", "rte", "cb", "wic", "qnli", "qqp", "mrpc"]:
            datasets["test"] = datasets["validation"]

        if self.task in ["mr", "cr", "subj", "SST-2", "trec",  "sst-5",
                         "boolq", "rte", "cb", "wic", "qnli", "qqp", "mrpc", 
                         "emotion", "enron_spam", "ag_news", "amazon_cf", "anketa"]:
            # First filter, then shuffle, otherwise this results in a bug.
            # Samples `num_samples` elements from train as training and development sets.
            sampled_train = []
            sampled_dev = []
            for label in range(self.num_labels):
                data = shuffled_train.filter(lambda example: int(example['label']) == label)
                print(label, np.unique(data["label"]))
                num_samples = min(len(data)//2, self.num_samples)
                print(num_samples)
                sampled_train.append(data.select([i for i in range(num_samples)]))
                sampled_dev.append(data.select([i for i in range(num_samples, num_samples*2)]))

            # Joins the sampled data per label.
            datasets["train"] = concatenate_datasets(sampled_train)
            datasets["validation"] = concatenate_datasets(sampled_dev)
        return datasets

    def get_datasets(self):
        datasets = self.load_datasets()
        if self.num_samples is not None:
            datasets = self.sample_datasets(datasets)
            datasets = self.post_processing(datasets)
            label_distribution_train = Counter(datasets["train"]["label"])
            label_distribution_dev = Counter(datasets["validation"]["label"])
        return datasets 


class MR(AbstractTask):
    task = "mr"
    num_labels = 2 
    metric = [metrics.accuracy]

    def load_datasets(self):
        dataset_args = {}
        print("task ", self.task)
        data_dir = join(self.data_dir, self.task) 
        data_files = {
            "train": join(data_dir, "train.json"),
            "test": join(data_dir, "test.json")
            }        
        return load_dataset("json", data_files=data_files, cache_dir=self.cache_dir, **dataset_args)


class CR(MR):
    task = "cr"
    num_labels = 2 
    metric = [metrics.accuracy]
   
class Subj(MR):
    task = "subj"
    num_labels = 2 
    metric = [metrics.accuracy]
   
class SST2(MR):
    task = "SST-2"
    num_labels = 2 
    metric = [metrics.accuracy]
    
class Trec(MR):
    task = "trec"
    num_labels = 6 
    metric = [metrics.accuracy]
    
class SST5(MR):
    task = "sst-5"
    num_labels = 5
    metric = [metrics.accuracy]
    
class BoolQ(AbstractTask):
    task = "boolq"
    num_labels = 2
    labels_list = ['0', '1'] # [False, True]
    metric = [metrics.accuracy]
    
    def load_datasets(self):
        return load_dataset('super_glue', self.task, script_version="master")
        

class RTE(BoolQ):
    task = "rte"
    num_labels = 2
    labels_list = ['0', '1'] # [entailment, not_entailment]
    metric = [metrics.accuracy]

class CB(BoolQ):
    task = "cb"
    num_labels = 3
    labels_list = ['0', '1', '2'] # entailment, contradiction, neutral
    metric = [metrics.accuracy, metrics.f1_macro]


class WiC(BoolQ):
    task = "wic"
    metric = [metrics.accuracy]
    labels_list = ['0', '1']
    num_labels = 2 


class QQP(AbstractTask):
    task = "qqp"
    num_labels = 2
    labels_list = ['0', '1'] # ["not_duplicate", "duplicate"]
    metric = [metrics.accuracy, metrics.f1]
    
    def load_datasets(self):
        return load_dataset('glue', self.task, script_version="master")

class QNLI(QQP):
    task = "qnli"
    num_labels = 2
    labels_list = ['0', '1'] # ["entailment", "not_entailment"]
    metric = [metrics.accuracy]


class MRPC(QQP):
    task = "mrpc"
    num_labels = 2
    labels_list = ['0', '1'] # ["not_equivalent", "equivalent"]
    metric = [metrics.accuracy, metrics.f1]

class Emotion(AbstractTask):
    task = "emotion"
    num_labels = 6
    labels_list = ['0', '1', '2', '3', '4', '5']
    metric = [metrics.accuracy]
    
    def load_datasets(self):
        dsets = load_dataset('SetFit/emotion')
        for split, dset in dsets.items():
            dsets[split] = dset.rename_column("text", "source")
        return dsets

class EnronSpam(AbstractTask):
    task = "enron_spam"
    num_labels = 2
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    
    def load_datasets(self):
        dsets = load_dataset('SetFit/enron_spam')
        for split, dset in dsets.items():
            dsets[split] = dset.rename_column("text", "source")
        return dsets

class AGNews(AbstractTask):
    task = "ag_news"
    num_labels = 4
    labels_list = ['0', '1', '2', '3']
    metric = [metrics.accuracy]
    
    def load_datasets(self):
        dsets = load_dataset('SetFit/ag_news')
        for split, dset in dsets.items():
            dsets[split] = dset.rename_column("text", "source")
        return dsets

class AmazonCF(AbstractTask):
    task = "amazon_cf"
    num_labels = 2
    labels_list = ['0', '1']
    metric = [metrics.mcc]
    
    def load_datasets(self):
        dsets = load_dataset('SetFit/amazon_counterfactual_en')
        for split, dset in dsets.items():
            dsets[split] = dset.rename_column("text", "source")
        return dsets

class Anketa(AbstractTask):
    task = "anketa"
    num_labels = 3 
    labels_list = ['0', '1', '2']
    metric = [metrics.accuracy, metrics.f1_macro]
    
    def load_datasets(self):

        data_train = pd.read_csv('data_files/data_train_192.csv',sep=',')
        data_test = pd.read_csv('data_files/data_test_192.csv', sep = ',')
        
        data_train['class_label'] = np.where(data_train['class_label'] == -1, 2, data_train['class_label'])
        data_test['class_label'] = np.where(data_test['class_label'] == -1, 2, data_test['class_label'])
        train_text,train_label = data_train['comment'], data_train['class_label']
        test_text,test_label = data_test['comment'], data_test['class_label']
    
        dsets = {'train':Dataset.from_dict({'source':train_text, 'label':train_label}),
            'test':Dataset.from_dict({'source':test_text, 'label':test_label})
        }
        
        return dsets

class AnketaBinary(AbstractTask):
    task = "anketa_binary"
    num_labels = 2 
    labels_list = ['0', '1']
    metric = [metrics.accuracy, metrics.f1_macro]
    
    def load_datasets(self):

        data_train = pd.read_csv('data_files/data_train_128_2.csv',sep=',')
        data_test = pd.read_csv('data_files/data_test_128_2.csv', sep = ',')
        
        data_train['class_label'] = np.where(data_train['class_label'] == -1, 0, data_train['class_label'])
        data_test['class_label'] = np.where(data_test['class_label'] == -1, 0, data_test['class_label'])
        train_text,train_label = data_train['comment'], data_train['class_label']
        test_text,test_label = data_test['comment'], data_test['class_label']
    
        dsets = {'train':Dataset.from_dict({'source':train_text, 'label':train_label}),
            'test':Dataset.from_dict({'source':test_text, 'label':test_label})
        }
        
        return dsets


TASK_MAPPING = OrderedDict(
    [
        ('mr', MR),
        ('cr', CR),
        ('subj', Subj),
        ('trec', Trec),
        ('SST-2', SST2),
        ('sst-5', SST5),
        # superglue datasets.
        ('boolq', BoolQ),
        ('rte', RTE),
        ('cb', CB),
        ('wic', WiC),
        # glue datasets
        ('qqp', QQP),
        ('qnli', QNLI),
        ('mrpc', MRPC),
        # SetFit datasets
        ('emotion', Emotion), # Doesn't work with SetFit/emotion for some reason
        ("enron_spam", EnronSpam),
        ("ag_news", AGNews),
        ("amazon_cf", AmazonCF),
        # My datasets
        ("anketa", Anketa),
        ("anketa", AnketaBinary),
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, data_seed, num_samples, cache_dir, data_dir=None):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](
                data_seed=data_seed, 
                num_samples=num_samples, 
                cache_dir=cache_dir, 
                data_dir=data_dir)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
