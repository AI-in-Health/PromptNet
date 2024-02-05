import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# import clip
import json
import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image

if not os.path.exists('prompt/'):
    os.makedirs('prompt/')

ann_path = 'promptreports/promptreport.json'
ann = json.loads(open(ann_path, 'r').read())
processor = MedCLIPProcessor()
image = Image.open('./example_data/view1_frontal.jpg')
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.cuda()
embed_text=[]
for example in ann:
    inputs = processor(
        text=[example['reports']],
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        text_embding=outputs['text_embeds']
    embed_text.append(text_embding)


embding = torch.stack(embed_text, dim=0) #2056*1*512

torch.save(embding, 'prompt/prompt.pth')
# print('text_features', embding.size())
#    temp = {'reports_ids': example['reports_ids'], 'embd_reports': text_features}
#    embed_report.append(temp)
# resFilet = 'embding/embdreport.json'
# json.dump(embed_report, open(resFilet , 'w'))











