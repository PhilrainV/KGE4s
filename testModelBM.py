import sys
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from LLMEJ1 import LLMJEmodel

def findREpos1(NerHead, Nertail):
    try:
        headPos = torch.where(NerHead.argmax(dim=2) == 1)[1]
        headNum = torch.where(NerHead.argmax(dim=2) == 1)[0]
        tailPos = torch.where(Nertail.argmax(dim=2) == 1)[1]
        tailNum = torch.where(Nertail.argmax(dim=2) == 1)[0]
    except:
        print(NerHead)
        return []

    NERheadGrather = [[] for i in range(NerHead.shape[0])]
    for num in range(NerHead.shape[0]):
        try:
            headSinglePos = headPos[headNum == num].tolist()
            tailSinglePos = tailPos[tailNum == num].tolist()
            tailStartPosNum = 0
            for headPosNum in range(len(headSinglePos)):
                for tailPosNum in range(tailStartPosNum, len(tailSinglePos)):
                    if headSinglePos[headPosNum] <= tailSinglePos[tailPosNum]:
                        NERheadGrather[num].append([headSinglePos[headPosNum], tailSinglePos[tailPosNum]])
                        tailStartPosNum = tailPosNum
                        break
        except:
            continue

    return NERheadGrather

def LoadTestData(dataPath):
    with open(dataPath, 'r', encoding='utf-8') as trainDataFile:
        testLines = trainDataFile.readlines()

    testAllData = []

    for i in tqdm(range(len(testLines))):
        line = testLines[i]
        linedict = json.loads(line)
        contextSingle = linedict['sentText']
        NER_Gather = linedict['relationMentions']
        RE_Gather = []
        for Ner_gatherSingle in NER_Gather:
            Ner1TextSingle = Ner_gatherSingle['em1Text']
            Ner2TextSingle = Ner_gatherSingle['em2Text']
            RE_Gather.append([Ner1TextSingle, Ner2TextSingle])
        testAllData.append([contextSingle, RE_Gather])

    return testAllData


def computeModedl(enconderToken, testModel, allTestData, device):
    intersectionNum ,accNum, RecallNum, F1 = 0, 0, 0,0

    for n in range(len(allTestData)):
        context = allTestData[n][0]
        label = allTestData[n][1]
        inputEncoder = enconderToken.batch_encode_plus(batch_text_or_text_pairs=[context],
                                                       truncation=True,
                                                       padding=True,
                                                       max_length=100,
                                                       return_tensors='pt')
        inputEncoder = inputEncoder.to(device)
        preNerHead, preNerTail, preNERheadGrather, preREhead, preREtail, attPos,_ = testModel(inputEncoder,0,0,0)
        predREgather = []
        NER2PosGrather = findREpos1(preREhead, preREtail)

        for i in range(len(preNERheadGrather[0])-1):
            preNER1single = preNERheadGrather[0][i]
            NER1 = context[preNER1single[0]-1:preNER1single[1]]
            for NER2Pos in NER2PosGrather[i]:
                if NER2Pos !=[]:
                    NER2 = context[NER2Pos[0]-1:NER2Pos[1]]
                    if NER1 != NER2:
                        predREgather.append([NER1,NER2])

        for singPredRE in predREgather:
            if singPredRE in label:
                intersectionNum +=1
        RecallNum +=len(label)
        accNum += len(predREgather)

    Acc = intersectionNum /accNum
    Recall = intersectionNum /RecallNum
    F1 = 2*Acc*Recall/(Recall+Acc)

    return Acc, Recall, F1


if __name__ == '__main__':
    # testDataPath = './data/testData.json'
    testDataPath = './data/ATMtestReLabel.json'
    # testDataPath = './data/ATMtrainReLabel.json'
    enconderModel = 'hfl/rbt6'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    allTestData = LoadTestData(testDataPath)
    pretrained = AutoModel.from_pretrained(enconderModel)
    enconderToken = AutoTokenizer.from_pretrained(enconderModel)
    pretrained = pretrained.to(device)
    testModel = LLMJEmodel(pretrained)
    CasrelState_dict = torch.load('model/LLMJE1(EXCL)95.pth', map_location='cuda:0')   #JPEA40.pth
    testModel.load_state_dict(CasrelState_dict['LLMJEmodel'])
    testModel.to(device)
    Acc, Recall, F1 = computeModedl(enconderToken, testModel, allTestData, device)
    print('Acc:', Acc, '   Recall:', Recall, '   F1:', F1)
