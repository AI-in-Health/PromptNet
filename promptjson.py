#generate prompt text and corresponding ids
import json
import re
import os
import random

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report


if not os.path.exists('promptreports/'):
    os.makedirs('promptreports/')
result_report=[]
ann_path='data/mimic_cxr/annotation.json'
ann = json.loads(open(ann_path, 'r').read())
for example in ann['train']:
    tokens = clean_report_mimic_cxr(example['report'])
    id=example['id']
    # split=clean_report_iu_xray(example['split'])
    temp = {'reports_ids': id, 'reports': tokens}
    result_report.append(temp)

# random.seed(100)
#selected 10000 reports randomly
random_report=random.sample(result_report, 10000)

resFilet = 'promptreports/promptreport.json'
json.dump(random_report, open(resFilet , 'w'))



