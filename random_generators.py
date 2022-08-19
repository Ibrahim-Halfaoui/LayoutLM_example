import random
import time
import pandas as pd
from datetime import datetime
from collections import Counter
from faker import Factory
import geonamescache
import json
from transliterate import translit
import exrex
from PIL import Image,ImageFont,ImageDraw
from matplotlib import pyplot as plt
CYRILLIC_FONT_FILE = '/home/stefan.kokov/temp/Alice-Regular.ttf'
##### this can be put in some other file, regex for all european vats
vats = '''^((AT)?U[0-9]{8}|(BE)?0[0-9]{9}|(BG)?[0-9]{9,10}|(CY)?[0-9]{8}L|
(CZ)?[0-9]{8,10}|(DE)?[0-9]{9}|(DK)?[0-9]{8}|(EE)?[0-9]{9}|
(EL|GR)?[0-9]{9}|(ES)?[0-9A-Z][0-9]{7}[0-9A-Z]|(FI)?[0-9]{8}|
(FR)?[0-9A-Z]{2}[0-9]{9}|(GB)?([0-9]{9}([0-9]{3})?|[A-Z]{2}[0-9]{3})|
(HU)?[0-9]{8}|(IE)?[0-9]S[0-9]{5}L|(IT)?[0-9]{11}|
(LT)?([0-9]{9}|[0-9]{12})|(LU)?[0-9]{8}|(LV)?[0-9]{11}|(MT)?[0-9]{8}|
(NL)?[0-9]{9}B[0-9]{2}|(PL)?[0-9]{10}|(PT)?[0-9]{9}|(RO)?[0-9]{2,10}|
(SE)?[0-9]{12}|(SI)?[0-9]{8}|(SK)?[0-9]{10})$'''
####
vats=vats.replace('\n','')

#### medicine list, unbeliavably stupid format
medicine_file = r'/data/uniqa_samples-todor/custom/Prilogenie-1-Export-202208051649.xlsx'
medicine = pd.read_excel(medicine_file)
medicine = medicine.iloc[10:,1:6].fillna('').sum(axis=1).tolist()



#################### all the local function are of format get_LABEL_NAME and should generate only 1 string
def show_img(img, cmap='gray', title='title'):
    plt.figure(figsize=(8, 12))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()

def get_item_name():
    return translit(random.choice(medicine),'bg')

def get_from_id():
    return exrex.getone(vats)

def get_to_id():
    return exrex.getone(vats)

def get_from_address():
    fake = Factory.create('bg_BG')
    _from_address=translit(fake.address(),'bg')
    _from_address = _from_address.replace('\n',' ')
    return _from_address


def get_to_address():
    fake = Factory.create('bg_BG')
    _to_address=translit(fake.address(),'bg')
    _to_address = _to_address.replace('\n',' ')
    return _to_address

def get_to_name():
    fake = Factory.create('bg_BG')
    return fake.name()

def get_from_name():
    fake = Factory.create('bg_BG')
    return fake.name()

def get_number():
    return str(random.randint(999999, 9999999999))

def get_to_postcode():
    return str(random.randint(1000, 999999))

def get_from_postcode():
    return str(random.randint(1000, 999999))

def get_to_city():
    gc = geonamescache.GeonamesCache()
    _cities = gc.get_cities()
    return translit(list(_cities.values())[random.randint(0,15000)]['name'],'bg')

def get_from_city():
    gc = geonamescache.GeonamesCache()
    _cities = gc.get_cities()
    return translit(list(_cities.values())[random.randint(0,15000)]['name'],'bg')


def get_date():
    formats = ['%Y/%m/%d','%Y-%m-%d','%m-%d-%Y','%m/%d/%Y','%d-%m-%Y','%d/%m/%Y']
    _d = random.randint(1, int(time.time()))
    return datetime.fromtimestamp(_d).strftime(random.choice(formats))


def get_item_value():
    return str(round(random.uniform(0.1, 200),2))

def get_total():
    return str(round(random.uniform(0.1, 9999),2))

def get_vat_percentage():
    return str(round(random.uniform(0.1, 20))) +'%'

def get_vat_amount():
    return str(round(random.uniform(0.1, 1000), 1))


if __name__ == '__main__':
    #### labels_dir and img_dir are for test purposes
    labels_dir = r'/data/mixed-invoices/labels_ocr/Vivacom_0029319703_15.07.22.pdf.json'
    dic = json.load(open(labels_dir))
    labels = []
    bboxes = []
    bboxes_ratios =[]
    for item in dic:
        if item['label'] != 'other':
            labels.append(item['label']+(str(item['group']) if item['group'] else 'n' ))
            bboxes.append(item['bbox'])
            bboxes_ratios.append(len(item['text']))
    ### counters, how many times does a label appear and whats the sum of all len() that have a given label
    number_of_occ = Counter(labels)
    sum_of_len = Counter(list(number_of_occ))
    ### filling the counters
    for i in sum_of_len:
        sum_of_len[i] = 0
    for k,v in zip(labels,bboxes_ratios):
        sum_of_len[k] += v
    ### generate fields, call all local functions get_LABEL_NAME for labels present in the file
    generation = dict.fromkeys(set(labels))
    for k,v in generation.items():
        try:
            generation[k] = locals()[f'get_{k[8:-1]}']()
        except:
            continue

    ### if generated item is > 1.4*len of all bboxes for that item, slice it to random slice between 0.8 and 1.2
    ### the idea here being that we dont want anything too large to fit in our Bounding boxes as they are relatively small usually
    ### can be played with the ratios, usually affects only address and sometimes item_name
    for k,v in generation.items():
        if generation[k]:
            if len(generation[k]) > round(1.4*(sum_of_len[k])):
                generation[k] = v[:round(((random.uniform(0.8, 1.2)))*(sum_of_len[k]))]

    new_text =[]
    new_label= []
    new_bbox =[]

    ###counter for the loop
    new_counter = Counter(list(number_of_occ))
    for i in new_counter:
        new_counter[i] =0
    old_ratio = dict.fromkeys(set(labels))
    ### fills the bounding boxes with the new text with the methodology, return new B-,I-,E- labels
    ### important - the new labels are correct, but different item_names and values cannot be derived from new_labels, the first word of
    ### item_name will always have B_invoice_item_name value

    ### everywhere [:-1] exists is to fit the solution for multiple item names and values, see the keys in generation dictionary and line 118
    for l,bb,r in zip(labels,bboxes,bboxes_ratios):
            new_counter[l] += 1
            new_label.append('B_' + l[:-1] if new_counter[l] == 1 else 'I_' + l[-1] if new_counter[l] != number_of_occ[l] else 'E_' + l[:-1])
            old_ratio[l] = 0 if new_counter[l] == 1 else old_ratio[l]
            new_text.append(generation[l][old_ratio[l]:round((r/sum_of_len[l])*len(generation[l]))+old_ratio[l]])
            old_ratio[l] = round((r/sum_of_len[l])*len(generation[l])) + old_ratio[l]
            new_bbox.append(bb)
    ###########drawing
    img_dir = r'/data/mixed-invoices/data/Vivacom_0029319703_15.07.22.pdf.png'
    im=Image.open(img_dir)
    im2 = im.copy()
    draw = ImageDraw.Draw(im)
    ### new_bbox here is meaningless, its the original bboxes we derive from labels
    for nt,bbx,nl in zip(new_text,new_bbox,new_label):
        draw.rectangle((bbx),fill='white',outline='red')
        ### ups the size of the font untill it can no longer fit in the W or the H of the bbox
        ### there might be some ideas here how to make it even better
        selected_size = 1
        for size in range(1, int(bbx[3]) - int(bbx[1])):
            arial = ImageFont.FreeTypeFont(CYRILLIC_FONT_FILE, size=size)
            left, top, right, bottom = arial.getbbox(nt)
            w = right - left
            h = bottom - top
            if w > (bbx[2]-bbx[0]) or h > (bbx[3]-bbx[1]):
                break
            selected_size = size
        arial = ImageFont.FreeTypeFont(CYRILLIC_FONT_FILE, size=selected_size)
        draw.text(((bbx[0]+bbx[2])/2, (bbx[3]+bbx[1])/2), nt, fill='black', anchor='mm', font=arial)
    show_img(im)
    #im.save(r'/data/uniqa_samples-todor/custom/'+'1.png')
