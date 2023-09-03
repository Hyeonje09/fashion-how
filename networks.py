import torch.nn as nn
import torchvision.models as models

class ResExtractor(nn.Module):

    def __init__(self, resnetnum='101', pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == '18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == '152':
            self.resnet = models.resnet152(pretrained=pretrained)

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        return self.model_front(x)


class Baseline_ResNet_emo(nn.Module):
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        self.encoder = ResExtractor('101')
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        # 정규화 및 드롭아웃 레이어 추가
        self.batch_norm = nn.BatchNorm2d(2048)  # 배치 정규화 레이어 추가
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 레이어 추가

        self.daily_linear = nn.Linear(2048, 7)
        self.gender_linear = nn.Linear(2048, 6)
        self.embel_linear = nn.Linear(2048, 3)

    def forward(self, x):
        
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        # 4D로 데이터 변환
        flatten = flatten.unsqueeze(2).unsqueeze(3)

        # 정규화 및 드롭아웃 레이어 적용
        flatten = self.batch_norm(flatten)
        flatten = self.dropout(flatten)
        flatten = flatten.view(flatten.size(0), -1)  # 입력 데이터를 평탄화

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel



if __name__ == '__main__':
    pass
