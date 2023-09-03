from dataset import ETRIDataset_emo
from networks import ResExtractor, Baseline_ResNet_emo

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.utils.data.distributed

from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    net = Baseline_ResNet_emo().to(DEVICE)
    trained_weights = torch.load('./models/Baseline_ResNet_emo/h_model_20.pkl', map_location=DEVICE)
    net.load_state_dict(trained_weights)
    
    # 아래 경로는 포함된 샘플(train set에서 추출)의 경로로, 실제 추론환경에서의 경로는 task.ipynb를 참고 바랍니다. 
    df = pd.read_csv('./Dataset/info_etri20_emotion_test_sample.csv') # 샘플 경로입니다. 
    val_dataset = ETRIDataset_emo(df, base_path='./Dataset/Test_sample/') # 샘플 경로입니다.
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    daily_gt_list = np.array([])
    daily_pred_list = np.array([])
    gender_gt_list = np.array([])
    gender_pred_list = np.array([])
    embel_gt_list = np.array([])
    embel_pred_list = np.array([])

    for j, sample in tqdm(enumerate(val_dataloader)):
        for key in sample:
            sample[key] = sample[key].to(DEVICE)
        out_daily, out_gender, out_embel = net(sample)

        daily_gt = np.array(sample['daily_label'].cpu())
        daily_gt_list = np.concatenate([daily_gt_list, daily_gt], axis=0)
        gender_gt = np.array(sample['gender_label'].cpu())
        gender_gt_list = np.concatenate([gender_gt_list, gender_gt], axis=0)
        embel_gt = np.array(sample['embel_label'].cpu())
        embel_gt_list = np.concatenate([embel_gt_list, embel_gt], axis=0)

        daily_pred = out_daily
        _, daily_indx = daily_pred.max(1)
        daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)

        gender_pred = out_gender
        _, gender_indx = gender_pred.max(1)
        gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)

        embel_pred = out_embel
        _, embel_indx = embel_pred.max(1)
        embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)

    daily_top_1, daily_acsa = get_test_metrics(daily_gt_list, daily_pred_list)
    gender_top_1, gender_acsa = get_test_metrics(gender_gt_list, gender_pred_list)
    embel_top_1, embel_acsa = get_test_metrics(embel_gt_list, embel_pred_list)
    print("------------------------------------------------------")
    print(
        "Daily:(Top-1=%.5f, ACSA=%.5f), Gender:(Top-1=%.5f, ACSA=%.5f), Embellishment:(Top-1=%.5f, ACSA=%.5f)" % (
            daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa))
    print("------------------------------------------------------")
    out = (daily_top_1 + gender_top_1 + embel_top_1) / 3
    print(out)
    
    return out 


def get_test_metrics(y_true, y_pred, verbose=True):
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / cnf_matrix.sum(axis=1)

    return top_1, cs_accuracy.mean()


if __name__ == '__main__':
    main()