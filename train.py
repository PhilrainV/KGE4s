import sys
import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from transformers import AutoModel
from dataPreprocessing import REselfDataLoader
from LLMEJgj import LLMJEmodel
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


def computeAcc(PredictedData, label):
    acc = (PredictedData == label).sum().item() / len(label)
    effectivePos = label == 1
    if len(label[effectivePos]) !=0:
        effectiveAcc = (PredictedData[effectivePos] == label[effectivePos]).sum().item() / len(label[effectivePos])
    else:
        effectiveAcc = 0
    return acc, effectiveAcc


def computeNerGatherAcc(predNer, Nerlabel):
    intersectionNum, GatherAccNum, GatherRecallNum = 0, 0, 0
    for i in range(len(predNer)):
        for singPrePos in predNer[i]:
            if singPrePos in Nerlabel[i]:
                intersectionNum += 1
        GatherAccNum += len(predNer[i])
        GatherRecallNum += len(Nerlabel[i])
    return intersectionNum / GatherAccNum, intersectionNum / GatherRecallNum


#  最好可以补充一个验证集
def train(selfTrainData, Epochs):
    testRE2 = 0
    lr = 3e-4
    log=[]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler_lr = StepLR(optimizer, step_size=60, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in tqdm(range(Epochs)):
        testi1,testi2 =0,0
        sumTrainLoss, sumTrainNerLoss, sumTrainRELoss = 0, 0, 0
        for step, (inputTextData, NerPosHeadLabel, NerPosTailLabel, RE1Pos, RE2PosLabel, context) in enumerate(
                selfTrainData):
            inputTextData, NerPosHeadLabel, NerPosTailLabel = inputTextData.to(device), NerPosHeadLabel.to(
                device), NerPosTailLabel.to(device)
            preNerHead, preNerTail, preNERheadGrather, preREhead, preREtail, attPos,RE2LabelIndex = model(inputTextData,RE1Pos,"GPU",1)

            NerPosHeadLabelFlatten, NerPosTailLabelFlatten = NerPosHeadLabel.reshape(-1), NerPosTailLabel.reshape(-1)
            backPosSelectFlatten = attPos.reshape(-1) == 1

            lossNer = criterion(preNerHead.reshape(-1, 2)[backPosSelectFlatten],
                                NerPosHeadLabelFlatten[backPosSelectFlatten]) + criterion(
                preNerTail.reshape(-1, 2)[backPosSelectFlatten], NerPosTailLabelFlatten[backPosSelectFlatten])


            preREheadLabel, preREtailLabel = torch.tensor([]).to(device), torch.tensor([]).to(device)
            labelattPos = torch.tensor([]).to(device)
            preREpot = len(RE2LabelIndex)
            # print(RE2LabelIndex)
            try:
                for RE2LabelIndexSingle in RE2LabelIndex:
                    BatchNum,RE2LabelNum =RE2LabelIndexSingle
                    preREheadLabel = torch.cat((preREheadLabel,RE2PosLabel[BatchNum][RE2LabelNum][0].to(device)),dim=0)
                    preREtailLabel = torch.cat((preREtailLabel,RE2PosLabel[BatchNum][RE2LabelNum][1].to(device)),dim=0)
                    labelattPos = torch.cat((labelattPos,attPos[BatchNum].to(device)),dim=0)
                    testRE2 =testRE2 +1
            except:
                print('------------------------')
                print(context)
                print(RE2LabelIndex)
                print(RE2PosLabel)
                print(preREheadLabel)
                print('=====================')
                print(preREheadLabel)
                print(RE2PosLabel[BatchNum][RE2LabelNum][0])
                print(testRE2)
                sys.exit(0)

            if len(RE2LabelIndex) != 0:
                preREheadLabel, preREtailLabel = preREheadLabel.long(), preREtailLabel.long()
                REbackPosSelectFlatten = (labelattPos == 1).reshape(-1)

            try:
                lossRE = (criterion(preREhead.reshape(-1, 2)[REbackPosSelectFlatten], preREheadLabel[REbackPosSelectFlatten]) \
                          + criterion(preREtail.reshape(-1, 2)[REbackPosSelectFlatten], preREtailLabel[REbackPosSelectFlatten])) / preREpot
                testi1 += 1
            except:
                testi2 +=1
                lossRE = torch.tensor(float('nan'))

            sumTrainNerLoss = sumTrainNerLoss + lossNer
            sumTrainRELoss = sumTrainRELoss + lossRE
            loss = lossRE + lossNer
            # loss = lossNer
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            sumTrainLoss = sumTrainLoss + loss

            if step % 10 == 0:
                # print(epoch, step, '   NERloss:', lossNer.item(), '   REloss:', lossRE.item(), '   loss:', loss.item())
                print(epoch, step, '   NERloss:', lossNer.item())

        scheduler_lr.step()
        trainNerLoss = sumTrainNerLoss.item() / (step + 1)
        if (torch.isnan(sumTrainRELoss) == False):
            trainRELoss = sumTrainRELoss.item() / (step + 1)
        else:
            trainRELoss = lossRE
        trainLoss = sumTrainLoss.item() / (step + 1)

        NerHeadAcc, NerHeadEffectiveAcc = computeAcc(preNerHead.argmax(dim=2).reshape(-1)[backPosSelectFlatten],
                                                     NerPosHeadLabelFlatten[backPosSelectFlatten])
        NerTailAcc, NerTailEffectiveAcc = computeAcc(preNerTail.argmax(dim=2).reshape(-1)[backPosSelectFlatten],
                                                     NerPosTailLabelFlatten[backPosSelectFlatten])


        try:
            NerGatherAcc, NerGatherRecall = NerGatherAcc(preNERheadGrather, RE1Pos)
        except:
            NerGatherAcc,NerGatherRecall=torch.nan,torch.nan

        # NerGatherAcc, NerGatherRecall = computeNerGatherAcc(preNERheadGrather, RE1Pos)

        if (torch.isnan(sumTrainRELoss) == False):
            REheadAcc, REheadEffAcc = computeAcc(preREhead.reshape(-1, 2)[REbackPosSelectFlatten].argmax(dim=1),
                                                 preREheadLabel[REbackPosSelectFlatten])
            REtailAcc, REtailEffAcc = computeAcc(preREtail.reshape(-1, 2)[REbackPosSelectFlatten].argmax(dim=1),
                                                 preREtailLabel[REbackPosSelectFlatten])
        else:
            REheadAcc, REheadEffAcc, REtailAcc, REtailEffAcc = torch.nan, torch.nan, torch.nan, torch.nan

        # trainRELoss = 'nan'
        print(' ')
        print(testi1,testi2)
        print('=====================================================')
        print('trainNERloss:', trainNerLoss, '   trainREloss:', trainRELoss, '   trainLoss:', trainLoss)
        print('-----------------------------------------------------')
        print('HeadAcc:', NerHeadAcc, '   HeadEffAcc:', NerHeadEffectiveAcc, '   TailAcc:', NerTailAcc, '   TailAcc:',
              NerTailEffectiveAcc)
        print('NerAcc:', (NerHeadAcc + NerTailAcc) / 2, 'EffNerAcc:', (NerHeadEffectiveAcc + NerTailEffectiveAcc) / 2)
        print('-----------------------------------------------------')
        print('NerGatherAcc:', NerGatherAcc, 'NerGatherRecall:', NerGatherRecall)
        print('REHeadAcc:', REheadAcc, '   REHeadEffAcc:', REheadEffAcc, '   RETailAcc:', REtailAcc, '   RETailAcc:',
              REtailEffAcc)
        print('REAcc:', (REheadAcc + REtailAcc) / 2, 'EffREAcc:', (REheadEffAcc + REtailEffAcc) / 2)
        print(' ')

        # 需要写一个记录log程序
        if ((epoch + 1) % 5 == 0 or epoch + 1 == Epochs)and epoch>30:
            torch.save({'LLMJEmodel': model.state_dict()}, './model/LLMJE' + str(epoch + 1) + '.pth')
        log.append([trainNerLoss,trainRELoss,trainLoss])

    # 需要写一个ply打印acc和loss的程序，并保存折线图


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasetPath = './data/ATMtrainReLabel1.json'
    enconderModel = 'hfl/rbt6'
    selfTrainDataInput = REselfDataLoader(datasetPath, enconderModel, batchSize=6)
    pretrained = AutoModel.from_pretrained(enconderModel)
    pretrained = pretrained.to(device)
    model = LLMJEmodel(pretrained)
    model.fine_tuneing(False)
    model = model.to(device)
    train(selfTrainDataInput, 100)
