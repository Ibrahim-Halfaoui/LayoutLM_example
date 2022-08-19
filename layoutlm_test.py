from layoutlm_train import model_load, get_labels, AttrDict
from transformers import LayoutLMv3FeatureExtractor,LayoutLMForTokenClassification  ,LayoutLMv2ForTokenClassification, LayoutLMv3Tokenizer
from transformers import LayoutLMv3ForTokenClassification, LayoutLMTokenizer,LayoutXLMTokenizer,LayoutLMv2Tokenizer,LayoutLMv2FeatureExtractor

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from tqdm import tqdm
from layout.unilm.layoutlm.layoutlm.data.funsd import FunsdDataset
from transformers import LayoutLMTokenizer, LayoutXLMTokenizer, LayoutLMv2Tokenizer,LayoutLMv2FeatureExtractor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
import os
import json

font = ImageFont.truetype('/home/stefan.kokov/temp/Alice-Regular.ttf', 30)


def iob_to_label(label):
    if label != 'O':
        return label
    else:
        return "other"


def random_color():
    return np.random.randint(0, 255, 3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': '/home/todor.mihov/layoutLM/uniqa-health/layout/layoutlm_test_data', #test txts
        'model_name_or_path': "microsoft/layoutxlm-base",
        'max_seq_length': 512,
        'model_type': 'layoutlm', }

args = AttrDict(args)

pad_token_label_id = CrossEntropyLoss().ignore_index

label_file = '/data/uniqa_samples-todor/layoutlm/labels.txt'  ##!! <--- required: path to set(all_possible_iob_tagging_we_have_in_data)
labels = get_labels(label_file)
label_map = {i: label for i, label in enumerate(labels)}

tokenizer = LayoutXLMTokenizer.from_pretrained(
    "microsoft/layoutxlm-base")

eval_dataset: FunsdDataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="test")
eval_dataloader = DataLoader(eval_dataset, batch_size=1)

### IMPORTANT!!!! - XLM and v2 take the images in BGR, v3 takes them in RGB
imagesfinal = []
img_dir = '/data/uniqa_samples-todor/layoutlm/test_set_imgs'
images = sorted(os.listdir(img_dir))
feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
for item in images:
    for item in images:
        im = Image.open(os.path.join(img_dir, item)).convert('RGB')
        R, G, B = im.split()
        new_image = Image.merge("RGB", (B, G, R))
        imagesfinal.append(new_image)


encoding = feature_extractor(imagesfinal, return_offsets_mapping = True, return_tensors="pt")

if __name__ == '__main__':

    # load the model and labels
    PATH = r'/data/uniqa_samples-todor/custom/layout_lm_models/layoutxlm_primary.pt'  ## path to saved model
    num_labels = len(labels)
    model = model_load(PATH, num_labels)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    zz = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels,image=torch.unsqueeze(encoding.data['pixel_values'][zz], dim=0).to(device))
            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )
            zz +=1

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(results)

    ###############visualization
    ### create dictionary of results

    filenames = [eval_dataloader.dataset.features[_].file_name for _ in range(len(preds_list))]
    label2color = {f'{label}': f'rgb({random_color()[0]},{random_color()[1]},{random_color()[2]})' for label in
                   label_map}
    viz_results = {f'{i}': defaultdict(list) for i in filenames}
    for _ in range(len(preds_list)):
        for id, token_pred, box in zip(eval_dataloader.dataset.features[_].input_ids, preds[_].squeeze().tolist(),
                                       eval_dataloader.dataset.features[_].actual_bboxes):
            if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id,
                                                                    tokenizer.sep_token_id,
                                                                    tokenizer.pad_token_id]):
                # skip prediction + bounding box
                continue

            else:
                viz_results[eval_dataloader.dataset.features[_].file_name]['pred'].append(token_pred)
                viz_results[eval_dataloader.dataset.features[_].file_name]['box'].append(box)

    ### draw the results
    original_image_dir = '/data/mixed-invoices/data'
    lblz = '/data/mixed-invoices/labels_ocr'
    output_image_dir = '/home/todor.mihov/layoutLM/uniqa-health/layout/viz_results'
    for key, value in viz_results.items():
        with Image.open(os.path.join(original_image_dir, key)) as im:
            im = im.convert('RGB')
            draw = ImageDraw.Draw(im)
            x = 0
            for prediction, box in zip(value['pred'], value['box']):
                predicted_label = iob_to_label(label_map[prediction])
                if predicted_label != 'other':
                    if predicted_label[2:] != x:
                        draw.rectangle(box, outline=label2color[f'{prediction}'])
                        draw.text((box[2] - 30, box[3] + 10), text=predicted_label[10:],
                                  fill=label2color[f'{prediction}'], font=font)
                    else:
                        draw.rectangle(box, outline=label2color[f'{prediction}'])
                x = predicted_label[2:]

        im.save(os.path.join(output_image_dir, 'XLMrand1'+key))
