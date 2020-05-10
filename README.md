# Reproduce ZeroDCE

Link to the paper: [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/abs/2001.06826)





### Download SICE dataset [here](https://github.com/csjcai/SICE) 

prepare dataset with `code/dataset.py`(you need to modify the path in the source file) or prepare the dataset yourself.



Here is the directory structure of the dataset I'm using.

```bash
$ tree part1-256 --filelimit=10     

part1-256
├── test-toy
│   ├── 318_3.JPG
│   ├── 332_1.JPG
│   ├── 340_1.JPG
│   ├── 345_1.JPG
│   ├── 353_3.JPG
│   └── 356_7.JPG
├── train [2421 entries exceeds filelimit, not opening dir]
└── val [600 entries exceeds filelimit, not opening dir]
```



### Inference with pre-trained model

```bash
# go to the code directory
cd code/

python demo.py --device=-1 --testDir=../data/part1-512/test-toy \
               --ckpt=../train-jobs/ckpt/8LE-3_best_model.pth \
               --output-dir=../demo-output
```

This will process the images in ``../data/part1-512/test-toy ` and save results to  `../demo-output`. Results including output from ZeroDCE, and simple gamma corrections.  



### Usage(Python>=3.6 is required as I used f-strings):

I use relative path throughout my code, so please follow the exact directories structure as shown the [next section](#file-structure).

The arguments are explained in `--help`, e.g., to get help for train.py, run `python train.py --help`.

**You can configure hyper-parameters used by a dictionary named `hp`in `train.py`.**

```bash
# go to the code directory
cd code/

# 512 here means image size, train/val loss will be saved to ../train-jobs/log
nohup python train.py --device=0 --baseDir=../data/part1-512 \
  --experiment=8LE --n_LE=8 --numEpoch=150 \
  --weights 2 4 2 5 &

python eval.py --device=0 --testDir=../data/part1-512/test-toy \
  --ckpt=../train-jobs/ckpt/8LE-3_best_model.pth
```

Visualization and sanity checks can be found in `Demo.ipynb`



### File Structure

You need to follow this directory structure as I use **relative** paths. Upon root directory, you need to create

*  a `code/` directory and put python files in it
* a `data/` directory and put subdirectory and data in it, considering modify `dataset.py` to your needs
* empty directories `train-jobs/log`,  `train-jobs/ckpt`, `train-jobs/evaluation` as log/checkpoing/results will be saved to them

![image-20200503001251677](docs/file-structure.png)

