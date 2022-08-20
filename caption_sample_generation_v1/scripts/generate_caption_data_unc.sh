#!/usr/bin/env bash

# $ cd pseudo_sample_generation
# $ bash scripts/generate_pseudo_data_unc.sh
# 原始 --vg_dataset_path ../data/image_data
# 鹏城 229服务器：--vg_dataset_path /hdd/lhxiao/pseudo-q/data  # 主要是使用 data目录下detection_results中物体检测结果和属性检测结果
# 北京 226服务器：--vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data

OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 0  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 1  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 2  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 3  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 4  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 5  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 6  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 7  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 8  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 9  --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 10 --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 11 --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 12 --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 13 --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 14 --topn 3 --each_image_query 20;
OMP_NUM_THREADS=4 python caption_generation.py --vg_dataset_path /data_SSD1/lhxiao/pseudo-q/data --vg_dataset unc --split_ind 15 --topn 3 --each_image_query 20;
# python ./utils/merge_file.py ../data/pseudo_samples/unc/top3_query6/ unc;
# python ./utils/post_process.py ../data/pseudo_samples/unc/top3_query6/unc/ unc;
python ./utils/merge_file.py /data_SSD1/lhxiao/pseudo-q/data/caption_gen/unc/top3_query20/ unc;
python ./utils/post_process.py /data_SSD1/lhxiao/pseudo-q/data/caption_gen/unc/top3_query20/unc/ unc;

