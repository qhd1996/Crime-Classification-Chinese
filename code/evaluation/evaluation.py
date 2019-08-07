#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from sklearn.metrics import classification_report,accuracy_score, balanced_accuracy_score

# tags = '个人隐私泄露 非法信息技术 买凶杀人 涉黄 暴恐 政治敏感 金融犯罪 涉赌 涉毒 涉枪 药品 其他'
# labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#
#
# def evaluate(prediction, target):
#     report = classification_report(target, prediction, labels=labels, target_names=tags.split(), digits=4, output_dict = False)
#     # print(report)
#     return report
#
# model_name = 'attention'
# pred_file = '../../data/output/' + model_name + '_prediction.csv'
# prediction = pd.read_csv(pred_file, encoding='utf-8')['prediction']
# target = pd.read_csv('../../data/test/test_data.csv', encoding='utf-8')['domain']
# report = evaluate(prediction, target)
#
# def report_to_csv(report, model_name):
#     lines = report.split('\n')
#     report_data = []
#     for line in lines[2:]:
#         row = {}
#         row_data = line.strip(' ').split('    ')
#         if row_data == ['']:
#             continue
#         if '' in row_data:
#             row_data.remove('')
#         print(row_data)
#         row['class'] = row_data[0].strip(' ')
#         row['precision'] = float(row_data[1].strip(' '))
#         row['recall'] = float(row_data[2].strip(' '))
#         row['f1_score'] = float(row_data[3].strip(' '))
#         row['support'] = float(row_data[4].strip(' '))
#         report_data.append(row)
#     df = pd.DataFrame.from_dict(report_data)
#     df.to_excel('../../data/output/' + model_name + '_report.xls', encoding='utf-8', index=False)
#     return
#
# report_to_csv(report,model_name)

tags = 'privacy infotech murder sex terrorism politics finance gambling drug gun medicine others'.split(' ')
tagscn = '个人隐私泄露 非法信息技术 买凶杀人 涉黄 暴恐 政治敏感 金融犯罪 涉赌 涉毒 涉枪 药品 其他'.split(' ')
label = 0
tag2label = {}
label2tag = {}
for tag in tags:
    tag2label[tag] = label
    label2tag[label] = tag
    label += 1
label = 0
tagcn2label = {}
label2tagcn = {}
for tag in tagscn:
    tagcn2label[tag] = label
    label2tagcn[label] = tag
    label += 1
writer = pd.ExcelWriter('../../data/output/model_report.xlsx')
models = 'bow cnn gru attention'.split(' ')
for model_name in models:
    prediction_file = '../../data/output/' + model_name + '_prediction.csv'
    prediction = pd.read_csv(prediction_file, encoding='utf-8')['prediction']
    target = pd.read_csv('../../data/test/test_data.csv', encoding='utf-8')['domain']
    pred_dict = {}
    pred_dict['pred_privacy'] = prediction[:188]
    pred_dict['pred_infotech'] = prediction[188:353]
    pred_dict['pred_murder'] = prediction[353:361]
    pred_dict['pred_sex'] = prediction[361:392]
    pred_dict['pred_terrorism'] = prediction[392:394]
    pred_dict['pred_politics'] = prediction[394:396]
    pred_dict['pred_finance'] = prediction[396:439]
    pred_dict['pred_gambling'] = prediction[439:463]
    pred_dict['pred_drug'] = prediction[463:477]
    pred_dict['pred_gun'] = prediction[477:483]
    pred_dict['pred_medicine'] = prediction[483:488]
    pred_dict['pred_others'] = prediction[488:517]

    tar_dict = {}
    tar_dict['target_privacy'] = target[:188]
    tar_dict['target_infotech'] = target[188:353]
    tar_dict['target_murder'] = target[353:361]
    tar_dict['target_sex'] = target[361:392]
    tar_dict['target_terrorism'] = target[392:394]
    tar_dict['target_politics'] = target[394:396]
    tar_dict['target_finance'] = target[396:439]
    tar_dict['target_gambling'] = target[439:463]
    tar_dict['target_drug'] = target[463:477]
    tar_dict['target_gun'] = target[477:483]
    tar_dict['target_medicine'] = target[483:488]
    tar_dict['target_others'] = target[488:517]

    sheet = pd.read_excel('../../data/output/model_results.xlsx', sheet_name=model_name)
    sheet['accuracy'] = 0.0
    sum = 0.0
    for tag in tags:
        pred_name = 'pred_' + tag
        target_name = 'target_' + tag
        accuracy = accuracy_score(tar_dict[target_name], pred_dict[pred_name])
        # accuracy = pred_dict[pred_name] / prediction.count()
        sum += accuracy * len(tar_dict[target_name])
        sheet.loc[sheet['class'] == label2tagcn[tag2label[tag]], 'accuracy'] = accuracy
    sheet.loc[sheet['class'] == 'macro avg', 'accuracy'] = accuracy_score(target, prediction)
    sheet.loc[sheet['class'] == 'micro avg', 'accuracy'] = balanced_accuracy_score(target, prediction)
    sheet.loc[sheet['class'] == 'weighted avg', 'accuracy'] = sum / len(target)
    sheet = sheet[['class', 'f1_score', 'precision', 'recall', 'accuracy','support']]
    sheet.to_excel(writer, model_name, encoding='utf-8')
writer.save()

