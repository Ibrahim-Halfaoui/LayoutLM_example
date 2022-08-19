from transformers import LayoutLMv3ForTokenClassification, LayoutLMTokenizer,LayoutXLMTokenizer,LayoutLMv2Tokenizer,LayoutLMv2FeatureExtractor
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from layout.unilm.layoutlm.layoutlm.data.funsd import FunsdDataset, InputFeatures ## we need only funsd from here
from transformers import LayoutLMv3FeatureExtractor,LayoutLMForTokenClassification  ,LayoutLMv2ForTokenClassification, LayoutLMv3Tokenizer
from transformers import AdamW
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from PIL import Image
import os

### sorry for messy imports
### train for XLM specifically, if model is changed multiple changes have to be make for .from_pretrained(string) and the tokenizer/model names

def model_load(PATH, num_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutxlm-base",num_labels=num_labels) ### thats how XLM is called
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels





# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
### IMPORTANT!!!! - XLM and v2 take the images in BGR, v3 takes them in RGB
imagesfinal = []
img_dir = '/data/uniqa_samples-todor/layoutlm/training_set_imgs'
images = sorted(os.listdir(img_dir))
feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
for item in images:
    im = Image.open(os.path.join(img_dir,item)).convert('RGB')
    R, G, B = im.split()
    new_image = Image.merge("RGB", (B, G, R))
    imagesfinal.append(new_image)


encoding = feature_extractor(imagesfinal, return_tensors="pt")
if __name__ == '__main__':
    pad_token_label_id = CrossEntropyLoss().ignore_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # required: location to set(all_possible_iob_tagging_we_have_in_data), if the current file is wrong, preprocess_primary has the code to generate it
    label_file = '/data/uniqa_samples-todor/layoutlm/labels.txt'
    labels = get_labels(label_file)
    num_labels = len(labels)
    label_map = {i: label for i, label in enumerate(labels)}

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later

    args = {'local_rank': -1,
            'overwrite_cache': True,
            'data_dir': '/home/todor.mihov/layoutLM/uniqa-health/layout/layoutlm_train_data', # training txts
            'model_name_or_path': 'microsoft/layoutxlm-base',
            'max_seq_length': 512,
            'model_type': 'layoutlm', }

    args = AttrDict(args)

    tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
    # current implementation works only with bath_size of 1 due to external feeding of image tensors
    train_dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1)


    batch = next(iter(train_dataloader))
    input_ids = batch[0][0]
    print(tokenizer.decode(input_ids)) ### our check if we didnt mess anything so far :)

    model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutxlm-base", num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    global_step = 0
    num_train_epochs = 200
    t_total = len(train_dataloader) * num_train_epochs  # total number of training steps

    # put the model in training mode
    model.train()
    for epoch in range(num_train_epochs):
        zz=0
        for batch in tqdm(train_dataloader, desc="Training"):

            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)


            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels,image=torch.unsqueeze(encoding.data['pixel_values'][zz], dim=0).to(device))
            loss = outputs.loss

            #print loss every 10 steps
            if global_step % 10 == 0:
                print(f"Loss after {global_step} / {t_total} steps: {loss.item()}")

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            zz +=1
    # save the model
    PATH = r'/data/uniqa_samples-todor/custom/layout_lm_models/layoutlmRAND_PRIM_WORD.pt'
    torch.save(model.state_dict(), PATH)


