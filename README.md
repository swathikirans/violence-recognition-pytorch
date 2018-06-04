#
Learning to Detect Violent Videos using Convolutional Long Short-Term Memory

*Experimental release*

The source code associated with the paper [Learning to Detect Violent Videos using Convolutional Long Short-Term Memory](https://arxiv.org/abs/1709.06531), published in AVSS-2017. 

#### Prerequisites
* Python 3.5
* Pytorch 0.3.1
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
![alt text](https://github.com/swathikirans/violence-recognition-pytorch/dataset_fmt.jpg "")

To cite our paper/code:

```
Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to detect violent videos using convolutional long short-term memory.
" Advanced Video and Signal Based Surveillance (AVSS), 2017 14th IEEE International Conference on. IEEE, 2017
```
