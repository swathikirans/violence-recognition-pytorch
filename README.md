#
Learning to Detect Violent Videos using Convolutional Long Short-Term Memory


The source code associated with the paper [Learning to Detect Violent Videos using Convolutional Long Short-Term Memory](https://arxiv.org/abs/1709.06531), published in AVSS-2017. (*Experimental release*) 

#### Prerequisites
* Python 3.5
* Pytorch 0.3.0
#### Running

```
python main-run-vr.py --numEpochs 100 \
--lr 1e-4 \
--stepSize 25 \
--decayRate 0.5 \
--seqLen 20 \
--trainBatchSize 16 \
--memSize 256 \
--evalInterval 5 \
--evalMode horFlip \
--numWorkers 4 \
--outDir violence \
--fightsDirTrain fightSamplesTrainDir \
--noFightsDirTrain noFightSamplesTrainDir \
--fightsDirTest fightSamplesTestDir \
--noFightsDirTest noFightSamplesTestDir
```

The images should be arranged in the following way:

![](https://github.com/swathikirans/violence-recognition-pytorch/blob/master/dataset_fmt.jpg)


To cite our paper/code:

```
@inproceedings{sudhakaran2017learning,
  title={Learning to detect violent videos using convolutional long short-term memory},
  author={Sudhakaran, Swathikiran and Lanz, Oswald},
  booktitle={Advanced Video and Signal Based Surveillance (AVSS), 2017 14th IEEE International Conference on},
  pages={1--6},
  year={2017},
  organization={IEEE}
}

```
