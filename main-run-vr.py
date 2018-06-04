import torch
import os
import glob
from spatial_transforms import (Compose, ToTensor, FiveCrops, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip, TenCrops, FlippedImagesTest, CenterCrop)
from makeDataset import *
from createModel import *
from tensorboardX import SummaryWriter
import sys
import argparse



def make_split(fights_dir, noFights_dir):
    imagesF = []
    for target in sorted(os.listdir(fights_dir)):
        d = os.path.join(fights_dir, target)
        if not os.path.isdir(d):
            continue
        imagesF.append(d)
    imagesNoF = []
    for target in sorted(os.listdir(noFights_dir)):
        d = os.path.join(noFights_dir, target)
        if not os.path.isdir(d):
            continue
        imagesNoF.append(d)
    Dataset = imagesF + imagesNoF
    Labels = list([1] * len(imagesF)) + list([0] * len(imagesNoF))
    NumFrames = [len(glob.glob1(Dataset[i], "*.jpg")) for i in range(len(Dataset))]
    return Dataset, Labels, NumFrames

def main_run(numEpochs, lr, stepSize, decayRate, trainBatchSize, seqLen, memSize,
             evalInterval, evalMode, numWorkers, outDir, fightsDir_train, noFightsDir_train,
             fightsDir_test, noFightsDir_test):

    train_dataset_dir_fights = fightsDir_train
    train_dataset_dir_noFights = noFightsDir_train
    test_dataset_dir_fights = fightsDir_test
    test_dataset_dir_noFights = noFightsDir_test

    trainDataset, trainLabels, trainNumFrames = make_split(train_dataset_dir_fights, train_dataset_dir_noFights)
    testDataset, testLabels, testNumFrames = make_split(test_dataset_dir_fights, test_dataset_dir_noFights)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    normalize = Normalize(mean=mean, std=std)
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])

    vidSeqTrain = makeDataset(trainDataset, trainLabels, trainNumFrames, spatial_transform=spatial_transform,
                                seqLen=seqLen)

    trainLoader = torch.utils.data.DataLoader(vidSeqTrain, batch_size=trainBatchSize,
                            shuffle=True, num_workers=numWorkers, pin_memory=True, drop_last=True)

    if evalMode == 'centerCrop':
        test_spatial_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])
        testBatchSize = 1
    elif evalMode == 'tenCrops':
        test_spatial_transform = Compose([Scale(256), TenCrops(size=224, mean=mean, std=std)])
        testBatchSize = 1
    elif evalMode == 'fiveCrops':
        test_spatial_transform = Compose([Scale(256), FiveCrops(size=224, mean=mean, std=std)])
        testBatchSize = 1
    elif evalMode == 'horFlip':
        test_spatial_transform = Compose([Scale(256), CenterCrop(224), FlippedImagesTest(mean=mean, std=std)])
        testBatchSize = 1

    vidSeqTest = makeDataset(testDataset, testLabels, testNumFrames, seqLen=seqLen,
    spatial_transform=test_spatial_transform)


    testLoader = torch.utils.data.DataLoader(vidSeqTest, batch_size=testBatchSize,
                            shuffle=False, num_workers=int(numWorkers/2), pin_memory=True)


    numTrainInstances = vidSeqTrain.__len__()
    numTestInstances = vidSeqTest.__len__()

    print('Number of training samples = {}'.format(numTrainInstances))
    print('Number of testing samples = {}'.format(numTestInstances))

    modelFolder = './experiments_' + outDir # Dir for saving models and log files
    # Create the dir
    if os.path.exists(modelFolder):
        print(modelFolder + ' exists!!!')
        sys.exit()
    else:
        os.makedirs(modelFolder)
    # Log files
    writer = SummaryWriter(modelFolder)
    trainLogLoss = open((modelFolder + '/trainLogLoss.txt'), 'w')
    trainLogAcc = open((modelFolder + '/trainLogAcc.txt'), 'w')
    testLogLoss = open((modelFolder + '/testLogLoss.txt'), 'w')
    testLogAcc = open((modelFolder + '/testLogAcc.txt'), 'w')


    model = ViolenceModel(mem_size=memSize)


    trainParams = []
    for params in model.parameters():
        params.requires_grad = True
        trainParams += [params]
    model.train(True)
    model.cuda()

    lossFn = nn.CrossEntropyLoss()
    optimizerFn = torch.optim.RMSprop(trainParams, lr=lr)
    optimScheduler = torch.optim.lr_scheduler.StepLR(optimizerFn, stepSize, decayRate)

    minAccuracy = 50

    for epoch in range(numEpochs):
        optimScheduler.step()
        epochLoss = 0
        numCorrTrain = 0
        iterPerEpoch = 0
        model.train(True)
        print('Epoch = {}'.format(epoch + 1))
        writer.add_scalar('lr', optimizerFn.param_groups[0]['lr'], epoch+1)
        for i, (inputs, targets) in enumerate(trainLoader):
            iterPerEpoch += 1
            optimizerFn.zero_grad()
            inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
            labelVariable = Variable(targets.cuda())
            outputLabel = model(inputVariable1)
            loss = lossFn(outputLabel, labelVariable)
            loss.backward()
            optimizerFn.step()
            outputProb = torch.nn.Softmax(dim=1)(outputLabel)
            _, predicted = torch.max(outputProb.data, 1)
            numCorrTrain += (predicted == targets.cuda()).sum()
            epochLoss += loss.data[0]
        avgLoss = epochLoss/iterPerEpoch
        trainAccuracy = (numCorrTrain / numTrainInstances) * 100
        print('Training: Loss = {} | Accuracy = {}% '.format(avgLoss, trainAccuracy))
        writer.add_scalar('train/epochLoss', avgLoss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        trainLogLoss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avgLoss))
        trainLogAcc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))

        if (epoch+1) % evalInterval == 0:
            model.train(False)
            print('Evaluating...')
            testLossEpoch = 0
            testIter = 0
            numCorrTest = 0
            for j, (inputs, targets) in enumerate(testLoader):
                testIter += 1
                if evalMode == 'centerCrop':
                    inputVariable1 = Variable(inputs.permute(1, 0, 2, 3, 4).cuda(), volatile=True)
                else:
                    inputVariable1 = Variable(inputs[0].permute(1, 0, 2, 3, 4).cuda(), volatile=True)
                labelVariable = Variable(targets.cuda(async=True), volatile=True)
                outputLabel = model(inputVariable1)
                outputLabel_mean = torch.mean(outputLabel, 0, True)
                testLoss = lossFn(outputLabel_mean, labelVariable)
                testLossEpoch += testLoss.data[0]
                _, predicted = torch.max(outputLabel_mean.data, 1)
                numCorrTest += (predicted == targets[0]).sum()
            testAccuracy = (numCorrTest / numTestInstances) * 100
            avgTestLoss = testLossEpoch / testIter
            print('Testing: Loss = {} | Accuracy = {}% '.format(avgTestLoss, testAccuracy))
            writer.add_scalar('test/epochloss', avgTestLoss, epoch + 1)
            writer.add_scalar('test/accuracy', testAccuracy, epoch + 1)
            testLogLoss.write('Test Loss after {} epochs = {}\n'.format(epoch + 1, avgTestLoss))
            testLogAcc.write('Test Accuracy after {} epochs = {}%\n'.format(epoch + 1, testAccuracy))
            if testAccuracy > minAccuracy:
                savePathClassifier = (modelFolder + '/bestModel.pth')
                torch.save(model, savePathClassifier)
                minAccuracy = testAccuracy
    trainLogAcc.close()
    testLogAcc.close()
    trainLogLoss.close()
    testLogLoss.close()
    writer.export_scalars_to_json(modelFolder + "/all_scalars.json")
    writer.close()
    return True

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--stepSize', type=int, default=25, help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.5, help='Learning rate decay rate')
    parser.add_argument('--seqLen', type=int, default=20, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=16, help='Training batch size')
    parser.add_argument('--memSize', type=int, default=256, help='ConvLSTM hidden state size')
    parser.add_argument('--evalInterval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--evalMode', type=str, default='horFlip', help='Evaluation mode', choices=['centerCrop', 'horFlip', 'fiveCrops', 'tenCrops'])
    parser.add_argument('--numWorkers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--outDir', type=str, default='violence', help='Output directory')
    parser.add_argument('--fightsDirTrain', type=str, default='./datasets/violent_flow/frames/fights', help='Directory containing training fight sequences')
    parser.add_argument('--noFightsDirTrain', type=str, default='./datasets/violent_flow/frames/noFights', help='Directory containing training non-fight sequences')
    parser.add_argument('--fightsDirTest', type=str, default='./datasets/violent_flow/frames/fights', help='Directory containing testing fight sequences')
    parser.add_argument('--noFightsDirTest', type=str, default='./datasets/violent_flow/frames/noFights', help='Directory containing testing non-fight sequences')
    args = parser.parse_args()

    numEpochs = args.numEpochs
    lr = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    memSize = args.memSize
    evalInterval = args.evalInterval
    evalMode = args.evalMode
    numWorkers = args.numWorkers
    outDir = args.outDir
    fightsDir_train = args.fightsDirTrain
    noFightsDir_train = args.noFightsDirTrain
    fightsDir_test = args.fightsDirTest
    noFightsDir_test = args.noFightsDirTest
    main_run(numEpochs, lr, stepSize, decayRate, trainBatchSize, seqLen, memSize,
             evalInterval, evalMode, numWorkers, outDir, fightsDir_train,
             noFightsDir_train, fightsDir_test, noFightsDir_test)

__main__()
