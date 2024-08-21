import json
import sys
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils import data


def encodeNerPosFind(Token, context, posGather):
    encodeContextGather = []
    for Ei in range(len(posGather)):
        for Ej in range(len(posGather[Ei])):
            encodeContextGather.append(context[:posGather[Ei][Ej]])
    encodeContextGather.append(context)

    encodeContext = Token.batch_encode_plus(batch_text_or_text_pairs=encodeContextGather,
                                            truncation=True,
                                            padding=True,
                                            return_tensors='pt')

    encodePosGatherNA = torch.sum(encodeContext['attention_mask'], dim=1) - 1
    contextLong = encodePosGatherNA[-1]
    encodePosGather = encodePosGatherNA[:4]
    return contextLong.tolist(), encodePosGather.tolist()


def dataProcessing(dataPath,DataPreBM, Token):
    with open(dataPath, 'r', encoding='utf-8') as trainDataFile:
        trainLines = trainDataFile.readlines()
    with open(DataPreBM, 'r', encoding='utf-8') as trainDataBMFile:
        trainLinesBM = trainDataBMFile.readlines()

    preprocessingAllData = []

    for i in tqdm(range(len(trainLines))):
        line = trainLines[i]
        linedict = json.loads(line)
        contextSingle = linedict['sentText']
        NER_gather = linedict['relationMentions']
        RE1Pos, RE2Pos, RE1PosSelect, RE2PosLabel = [], [], [], []
        NER1dict,NERdict,NERallPos,BMpreR = [],[],[],[]
        NerPosHeadSingleLabel = torch.zeros(0)
        NerPosTailSingleLabel = torch.zeros(0)
        for NER_gatherSingle in NER_gather:
            Ner1TextSingle, Ner2TextSingle = NER_gatherSingle['em1Text'], NER_gatherSingle['em2Text']
            Ner1PosSingle = json.loads(NER_gatherSingle['em1Pos'].replace('(', '[').replace(')', ']'))
            Ner2PosSingle = json.loads(NER_gatherSingle['em2Pos'].replace('(', '[').replace(')', ']'))
            contextSingleLong, encodePosGather = encodeNerPosFind(Token, contextSingle, [Ner1PosSingle, Ner2PosSingle])

            try:
                NerPosHeadSingleLabel[encodePosGather[0]] = 1
                NerPosHeadSingleLabel[encodePosGather[2]] = 1
                NerPosTailSingleLabel[encodePosGather[1] - 1] = 1
                NerPosTailSingleLabel[encodePosGather[3] - 1] = 1
            except:
                NerPosHeadSingleLabel = torch.zeros(contextSingleLong)
                NerPosTailSingleLabel = torch.zeros(contextSingleLong)
                NerPosHeadSingleLabel[encodePosGather[0]] = 1
                NerPosHeadSingleLabel[encodePosGather[2]] = 1
                NerPosTailSingleLabel[encodePosGather[1] - 1] = 1
                NerPosTailSingleLabel[encodePosGather[3] - 1] = 1

            if ([encodePosGather[0], encodePosGather[1] - 1] not in RE1Pos) and (NER_gatherSingle['label'] == '1'):       # 多分类时应当改进
                RE1Pos.append([encodePosGather[0], encodePosGather[1] - 1])
                NER1dict.append(Ner1TextSingle)
                RE2PosHeadLabelSingle = torch.zeros(contextSingleLong)
                RE2PosTailLabelSingle = torch.zeros(contextSingleLong)
                RE2PosHeadLabelSingle[encodePosGather[2]], RE2PosTailLabelSingle[encodePosGather[3] - 1] = 1, 1
                RE2PosLabel.append([RE2PosHeadLabelSingle, RE2PosTailLabelSingle])
                if [encodePosGather[2], encodePosGather[3] - 1] not in RE2Pos:
                    RE2Pos.append([encodePosGather[2], encodePosGather[3] - 1])

            elif ([encodePosGather[0], encodePosGather[1] - 1] in RE1Pos) and (NER_gatherSingle['label'] == '1'):
                NerTailPosLabelIndex = RE1Pos.index([encodePosGather[0], encodePosGather[1] - 1])
                RE2PosHeadLabelSingle, RE2PosTailLabelSingle = RE2PosLabel[NerTailPosLabelIndex]
                RE2PosHeadLabelSingle[encodePosGather[2]], RE2PosTailLabelSingle[encodePosGather[3] - 1] = 1, 1
                RE2PosLabel[NerTailPosLabelIndex] = RE2PosHeadLabelSingle, RE2PosTailLabelSingle
                if [encodePosGather[2], encodePosGather[3] - 1] not in RE2Pos:
                    RE2Pos.append([encodePosGather[2], encodePosGather[3] - 1])

            if [encodePosGather[0], encodePosGather[1] - 1] not in NERallPos:
                NERallPos.append([encodePosGather[0], encodePosGather[1] - 1])
                NERdict.append(Ner1TextSingle)
            if [encodePosGather[2], encodePosGather[3] - 1] not in NERallPos:
                NERallPos.append([encodePosGather[2], encodePosGather[3] - 1])
                NERdict.append(Ner2TextSingle)

        preBMresult = trainLinesBM[i]
        preBMresultLinedict = json.loads(preBMresult)
        preBMresultGather = eval(preBMresultLinedict['preREMentions'])

        NER2PosGatherSingleBMpre = torch.zeros([len(RE1Pos), 2,contextSingleLong])
        for preBMresultSingle in preBMresultGather:
            try:
                NER1dictIndex = NER1dict.index(preBMresultSingle[0][0])
            except:
                continue
            for BM_Ner2TextSingle in preBMresultSingle[1]:
                try:
                    NER2dictIndex = NERdict.index(BM_Ner2TextSingle)
                except:
                    continue
                if RE1Pos[NER1dictIndex][0] < NERallPos[NER2dictIndex][0]:
                    NER2PosGatherSingleBMpre[NER1dictIndex][0][NERallPos[NER2dictIndex][0]] = 1
                    NER2PosGatherSingleBMpre[NER1dictIndex][1][NERallPos[NER2dictIndex][1]] = 1

        preprocessingAllData.append([linedict['sentText'], NerPosHeadSingleLabel, NerPosTailSingleLabel, RE1Pos, RE2PosLabel,NER2PosGatherSingleBMpre])

    return preprocessingAllData


class collater():
    def __init__(self, token):
        self.token = token

    def __call__(self, selfData):
        context, NerPosHeadLabel, NerPosTailLabel, RE1Pos, RE2PosLabel = [i[0] for i in selfData], [i[1] for i in
                                                                                                    selfData], [i[2] for
                                                                                                                i in
                                                                                                                selfData], [
            i[3] for i in selfData], [i[4] for i in
                                      selfData]  # inputTextData = self.token.batch_encode_plus(batch_text_or_text_pairs=context,
        inputTextData = self.token.batch_encode_plus(batch_text_or_text_pairs=context,
                                                     truncation=True,
                                                     padding=True,
                                                     max_length=100,
                                                     return_tensors='pt')
        lens = inputTextData['input_ids'].shape[1]
        RE2PosLabelCP = []
        for i in range(len(NerPosHeadLabel)):
            labelslen = len(NerPosHeadLabel[i])
            RE2PosLabelCP.append([])
            if (lens > labelslen):
                NerPosHeadLabel[i] = torch.cat((NerPosHeadLabel[i], torch.zeros(lens - labelslen)))
                NerPosTailLabel[i] = torch.cat((NerPosTailLabel[i], torch.zeros(lens - labelslen)))
                for k in range(len(RE2PosLabel[i])):
                    RE2HeadPos, RE2TailPos = RE2PosLabel[i][k]
                    RE2HeadPosCP = torch.cat((RE2HeadPos, torch.zeros(lens - labelslen)))
                    RE2TailPosCP = torch.cat((RE2TailPos, torch.zeros(lens - labelslen)))
                    RE2PosLabelCP[i].append([RE2HeadPosCP, RE2TailPosCP])
            else:
                RE2PosLabelCP[i] = RE2PosLabel[i]

        for j in range(len(NerPosHeadLabel)):
            if j == 0:
                NerPosHeadLabelTF = torch.unsqueeze(NerPosHeadLabel[j], 0)
                NerPosTailLabelTF = torch.unsqueeze(NerPosTailLabel[j], 0)
            else:
                headLabel2V = torch.unsqueeze(NerPosHeadLabel[j], 0)
                tailLabel2V = torch.unsqueeze(NerPosTailLabel[j], 0)
                if headLabel2V.shape[1] > NerPosTailLabelTF.shape[1]:
                    NerPosHeadLabelTF = torch.cat((NerPosHeadLabelTF, torch.unsqueeze(headLabel2V[0, :-1], 0)), dim=0)
                    NerPosTailLabelTF = torch.cat((NerPosTailLabelTF, torch.unsqueeze(tailLabel2V[0, :-1], 0)), dim=0)
                elif headLabel2V.shape[1] < NerPosTailLabelTF.shape[1]:
                    NerPosHeadLabelTF = torch.cat((torch.unsqueeze(NerPosHeadLabelTF[0, :-1], 0), headLabel2V), dim=0)
                    NerPosTailLabelTF = torch.cat((torch.unsqueeze(NerPosTailLabelTF[0, :-1], 0), tailLabel2V), dim=0)
                else:
                    NerPosHeadLabelTF = torch.cat((NerPosHeadLabelTF, headLabel2V), dim=0)
                    NerPosTailLabelTF = torch.cat((NerPosTailLabelTF, tailLabel2V), dim=0)

        return inputTextData, NerPosHeadLabelTF.long(), NerPosTailLabelTF.long(), RE1Pos, RE2PosLabelCP, context


def REselfDataLoader(selfData,selfDataPreBM, enconderModel='hfl/rbt6', batchSize=3):
    enconderToken = AutoTokenizer.from_pretrained(enconderModel)
    selfDataset = dataProcessing(selfData,selfDataPreBM, enconderToken)
    print('成功')
    sys.exit(0)
    collate_fn = collater(token=enconderToken)
    selfDatasetLoader = data.DataLoader(dataset=selfDataset, batch_size=batchSize, collate_fn=collate_fn,
                                        shuffle=True,
                                        drop_last=True)
    return selfDatasetLoader


if __name__ == '__main__':
    selfDataPath = './data/ATMtrainReLabel.json'
    selfDataPreBMPath = './data/ATMtrainRELabelBM.json'
    enconderModel = 'hfl/rbt6'
    a = REselfDataLoader(selfDataPath,selfDataPreBMPath, batchSize=3)
    for i, (a1, a2, a3, a4, a5, a6) in enumerate(a):
        print(a6)
        print(a4)
        print(a5)
        break
