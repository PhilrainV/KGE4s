import sys

import torch
import math
from torch import nn
from transformers import AutoModel
from dataPreprocessing import REselfDataLoader


def findREpos1(NerHead, Nertail):
    headPos = torch.where(NerHead.argmax(dim=2) == 1)[1]
    headNum = torch.where(NerHead.argmax(dim=2) == 1)[0]
    tailPos = torch.where(Nertail.argmax(dim=2) == 1)[1]
    tailNum = torch.where(Nertail.argmax(dim=2) == 1)[0]
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


def findNERpos(NerHead, Nertail):
    headPos = torch.where(NerHead == 1)[1]
    headNum = torch.where(NerHead == 1)[0]
    tailPos = torch.where(Nertail == 1)[1]
    tailNum = torch.where(Nertail == 1)[0]
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


class LLMJEmodel(nn.Module):
    def __init__(self, pretrainedModel):
        super().__init__()
        self.tuneing = False
        self.pertrained = None
        self.pretrainedModel = pretrainedModel
        self.BiGru = nn.GRU(input_size=768,
                            hidden_size=512,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)
        self.Bi2Gru = nn.GRU(input_size=768,
                             hidden_size=384,
                             num_layers=1,
                             bidirectional=True,
                             batch_first=True)
        self.W_Q = nn.Linear(1024, 512, bias=False)
        self.W_K = nn.Linear(1024, 512, bias=False)
        self.W_V = nn.Linear(1024, 512, bias=False)
        self.RE_W_Q = nn.Linear(2304, 768, bias=False)
        self.RE_W_K = nn.Linear(2304, 768, bias=False)
        self.RE_W_V = nn.Linear(2304, 768, bias=False)
        self._norm_fact = 1 / math.sqrt(512)
        self.RE_norm_fact = 1 / math.sqrt(768)
        self.fc1head = nn.Linear(512, 512)
        self.fc1tail = nn.Linear(512, 512)
        self.fc2head = nn.Linear(512, 2)
        self.fc2tail = nn.Linear(512, 2)
        self.RE_fc1head = nn.Linear(768, 512)
        self.RE_fc1tail = nn.Linear(768, 512)
        self.RE_fc2head = nn.Linear(512, 2)
        self.RE_fc2tail = nn.Linear(512, 2)
        self.dropout1head = nn.Dropout(p=0.3)
        self.dropout1tail = nn.Dropout(p=0.3)
        self.dropout2head = nn.Dropout(p=0.2)
        self.dropout2tail = nn.Dropout(p=0.2)
        self.RE_dropout1head = nn.Dropout(p=0.3)
        self.RE_dropout1tail = nn.Dropout(p=0.3)
        self.RE_dropout2head = nn.Dropout(p=0.2)
        self.RE_dropout2tail = nn.Dropout(p=0.2)

    def selfAttention(self, attIn):
        q = self.W_Q(attIn)
        k = self.W_K(attIn)
        v = self.W_V(attIn)
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        attOut = torch.bmm(dist, v)
        return attOut

    def self_RE_Attention(self, REattIn):
        REq = self.RE_W_Q(REattIn)
        REk = self.RE_W_K(REattIn)
        REv = self.RE_W_V(REattIn)
        REdist = torch.bmm(REq, REk.transpose(1, 2)) * self.RE_norm_fact
        REdist = torch.softmax(REdist, dim=-1)
        REattOut = torch.bmm(REdist, REv)
        return REattOut

    def REmodel(self, preTrainOut, RE1Pos, RE1AddContext, contextSingleREnum,NerFeature):
        contextSingleEnconder = torch.clone(preTrainOut[contextSingleREnum])
        RE1Encoder = contextSingleEnconder[RE1Pos[0]:RE1Pos[1] + 1]
        RE1Feature = torch.sum(RE1Encoder, dim=0) / RE1Encoder.shape[0]
        UpRE1Feature = RE1Feature.repeat(RE1AddContext[contextSingleREnum].shape[0], 1)
        RE1AddContextSingle = torch.cat((RE1AddContext[contextSingleREnum], UpRE1Feature), dim=1)
        RE1AcontextANer = torch.cat((RE1AddContextSingle,NerFeature),dim=1)
        RE1AcontextANer = torch.unsqueeze(RE1AcontextANer, dim=0)
        selfREAttOut = self.self_RE_Attention(RE1AcontextANer)
        REfc1headOut = self.RE_fc1head(selfREAttOut)
        REfc1tailOut = self.RE_fc1tail(selfREAttOut)
        REfc1headOut = self.RE_dropout1head(REfc1headOut)
        REfc1tailOut = self.RE_dropout1tail(REfc1tailOut)
        REfc2headOut = self.RE_fc2head(REfc1headOut)
        REfc2tailOut = self.RE_fc2tail(REfc1tailOut)
        REfc2headOut = self.RE_dropout1head(REfc2headOut)
        REfc2tailOut = self.RE_dropout1tail(REfc2tailOut)
        return REfc2headOut, REfc2tailOut

    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in self.pretrainedModel.parameters():
                i.requires_grad = True
            self.pretrainedModel.train()
            self.pretrained = self.pretrainedModel
        else:
            for i in self.pretrainedModel.parameters():
                i.requires_grad_(False)
            self.pretrainedModel.eval()
            self.pretrained = None

    def forward(self, inputTextData, NerPosHeadLabel, NerPosTailLabel, Ner1Gather, trainModel='GPU', train=0):
        if self.tuneing:
            preTrainOut = self.pretrained(**inputTextData).last_hidden_state
        else:
            with torch.no_grad():
                preTrainOut = self.pretrainedModel(**inputTextData).last_hidden_state

        BiGruOut, BiGruHidden = self.BiGru(preTrainOut)
        selfAttOut = self.selfAttention(BiGruOut)
        fc1headOut = self.fc1head(selfAttOut)
        fc1tailOut = self.fc1tail(selfAttOut)
        fc1headOut = self.dropout1head(fc1headOut)
        fc1tailOut = self.dropout1tail(fc1tailOut)
        fc2headOut = self.fc2head(fc1headOut)
        fc2tailOut = self.fc2tail(fc1tailOut)
        fc2headOut = self.dropout2head(fc2headOut)
        fc2tailOut = self.dropout2tail(fc2tailOut)

        headOut = fc2headOut
        tailOut = fc2tailOut

        NERheadGrather = findREpos1(headOut, tailOut)
        RE2LabelIndex = []
        RE1AddContext, _ = self.Bi2Gru(preTrainOut)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        REpreHeadResult, REpreTailResult = torch.tensor([]), torch.tensor([])
        RE2FeatureGather = torch.zeros_like(preTrainOut)

        if train == 1:
            NerGather = findNERpos(NerPosHeadLabel, NerPosTailLabel)
            for contextSingleNum in range(len(NerGather)):
                for NERposSingle in NerGather[contextSingleNum]:
                    NerEncoder = preTrainOut[contextSingleNum, NERposSingle[0]:NERposSingle[1] + 1]
                    NerFeature = torch.sum(NerEncoder, dim=0) / NerEncoder.shape[0]
                    RE2FeatureGather[contextSingleNum, NERposSingle[0]:NERposSingle[1], :] = NerFeature
        else:
            NerGather = NERheadGrather
            for contextSingleNum in range(len(NerGather)):
                for NERposSingle in NerGather[contextSingleNum]:
                    NerEncoder = preTrainOut[contextSingleNum, NERposSingle[0]:NERposSingle[1] + 1]
                    NerFeature = torch.sum(NerEncoder, dim=0) / NerEncoder.shape[0]
                    RE2FeatureGather[contextSingleNum, NERposSingle[0]:NERposSingle[1], :] = NerFeature

        if trainModel == 'GPU':
            REpreHeadResult, REpreTailResult = REpreHeadResult.to(device), REpreTailResult.to(device)
            RE2FeatureGather =RE2FeatureGather.to(device)

        for contextSingleREnum in range(len(NERheadGrather)):
            if len(NERheadGrather[contextSingleREnum]) != 0:
                for RE1PosNum in range(len(NERheadGrather[contextSingleREnum])):
                    RE1Pos = NERheadGrather[contextSingleREnum][RE1PosNum]
                    if train == 1:
                        if RE1Pos in Ner1Gather[contextSingleREnum]:
                            RE2LabelIndex.append([contextSingleREnum, Ner1Gather[contextSingleREnum].index(RE1Pos)])
                            REfc1headOut, REfc1tailOut = self.REmodel(preTrainOut, RE1Pos, RE1AddContext,
                                                                      contextSingleREnum,RE2FeatureGather[contextSingleREnum])
                            REpreHeadResult = torch.cat((REpreHeadResult, REfc1headOut), dim=0)
                            REpreTailResult = torch.cat((REpreTailResult, REfc1tailOut), dim=0)
                    else:
                        REfc1headOut, REfc1tailOut = self.REmodel(preTrainOut, RE1Pos, RE1AddContext,
                                                                  contextSingleREnum,RE2FeatureGather[contextSingleREnum])
                        REpreHeadResult = torch.cat((REpreHeadResult, REfc1headOut), dim=0)
                        REpreTailResult = torch.cat((REpreTailResult, REfc1tailOut), dim=0)

        return headOut, tailOut, NERheadGrather, REpreHeadResult, REpreTailResult, inputTextData[
            'attention_mask'], RE2LabelIndex


if __name__ == '__main__':
    # device = torch.device("cpu")
    datasetPath = './data/ATMtrainRELabel.json'
    enconderModel = 'hfl/rbt6'
    pretrained = AutoModel.from_pretrained(enconderModel)
    testModel = LLMJEmodel(pretrained)
    RE_Epoch = 20
    a = REselfDataLoader(datasetPath, enconderModel, batchSize=2)
    for i, (a1, a2, a3, a4, a5, a6) in enumerate(a):
        print(a6)
        print(a4)
        b0, b1, b2, b3, b4, b5, b6 = testModel(a1, a2, a3, a4, 'CPU', 1)
        # print(b5)

        break
