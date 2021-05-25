# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read all data in IMDB and merge them to a csv file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
from absl import app
from absl import flags
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string("raw_data_dir", "", "raw data dir")
flags.DEFINE_string("output_dir", "", "output_dir")


def dump_raw_data(contents, file_path):
  with open(file_path, "w") as ouf:
    writer = csv.writer(ouf, delimiter="\t", quotechar="\"")
    for line in contents:
      writer.writerow(line)

def load_data_by_id(sub_set, id_path):
  with open(id_path) as inf:
    id_list = inf.readlines()
  contents = []
  for example_id in id_list:
    example_id = example_id.strip()
    label = example_id.split("_")[0]
    file_path = os.path.join(FLAGS.raw_data_dir, sub_set, label, example_id[len(label) + 1:])
    with open(file_path) as inf:
      st_list = inf.readlines()
      assert len(st_list) == 1
      st = st_list[0].strip()
      contents += [(st, label, example_id)]
  return contents


def load_all_data(sub_set):
  contents = []
  for label in ["pos", "neg", "unsup"]:
    data_path = os.path.join(FLAGS.raw_data_dir, sub_set, label)
    if not os.path.exists(data_path):
      continue
    for filename in os.listdir(data_path):
      file_path = os.path.join(data_path, filename)
      with open(file_path) as inf:
        st_list = inf.readlines()
        assert len(st_list) == 1
        st = st_list[0].strip()
        example_id = "{}_{}".format(label, filename)
        contents += [(st, label, example_id)]
  return contents


def main(_):
  # load train
  header = ["content", "label", "id"]
  contents = load_data_by_id("train", FLAGS.train_id_path)
  os.mkdir(FLAGS.output_dir)
  dump_raw_data(
      [header] + contents,
      os.path.join(FLAGS.output_dir, "train.csv"),
  )
  # load test
  contents = load_all_data("test")
  dump_raw_data(
      [header] + contents,
      os.path.join(FLAGS.output_dir, "test.csv"),
  )


if __name__ == "__main__":
  # load train
  X_train = pd.read_csv('mex_train.txt',sep='\r\n', engine='python', header=None).loc[:,0]
  y_train = pd.read_csv('mex_train_labels.txt',sep='\r\n', engine='python', header=None).loc[:,0]


  t = {'s': X_train,
        'l': y_train
        }

  df = pd.DataFrame(t, columns= ['s', 'l'])
  df.to_csv ('train.csv', index = False, header=False,sep="\t")

  X_test = pd.read_csv('mex_val.txt',sep='\r\n', engine='python', header=None).loc[:,0]
  y_test = pd.read_csv('mex_val_labels.txt',sep='\r\n', engine='python', header=None).loc[:,0]

  v = {'s': X_test,
        'l': y_test
        }

  df = pd.DataFrame(v, columns= ['s', 'l'])
  df.to_csv ('test.csv', index = False, header=False,sep="\t")
