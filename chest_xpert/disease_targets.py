"""Conservative weak labels from curated MeSH terms and clinical assertions.

Missing findings are unknown, not automatically negative. These are weak
training labels, not independently adjudicated diagnoses.
"""
import re

DISEASES=['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural Thickening','Pneumonia','Pneumothorax']
PATTERNS=[r'atelecta(?:sis|ses|tic)',r'cardiomegaly|cardiac enlargement|heart.{0,12}enlarged',r'consolidat\w*',r'(?:pulmonary )?edema',r'(?:pleural )?effusion\w*|pleural fluid',r'emphysema\w*',r'(?:pulmonary )?fibrosis|fibrotic',r'hernia\w*',r'infiltrat\w*',r'mass(?:es)?',r'nodul\w*',r'pleural thicken\w*|thicken\w*[/ ]pleura',r'pneumonia\w*',r'pneumothora(?:x|ces)']
NEG=re.compile(r'\b(?:no|not|without|absent|negative for|free of|clear of|resolution of|resolved)\b',re.I)
UNCERTAIN=re.compile(r'\b(?:possible|possibly|may|might|could|suspect\w*|cannot exclude|can not exclude|versus|vs\.?|question\w*|if|evaluate for|rule out|concern for|concerning for)\b',re.I)


def disease_targets(row):
    target=[0.0]*len(DISEASES);mask=[0.0]*len(DISEASES)
    terms=[term.lower() for term in row.get('mesh_terms',[])]
    normal=any(term.strip()=='normal' for term in terms)
    if normal:mask=[1.0]*len(DISEASES)
    sections=row.get('target',{}).get('sections',[])
    text=' '.join(s['text'] for s in sections if s['heading'].lower() in {'findings','impression'})
    sentences=re.split(r'[.;]|\bbut\b|\bhowever\b',text.lower())
    for i,pattern in enumerate(PATTERNS):
        if any(re.search(r'\b(?:'+pattern+r')\b',term) for term in terms):
            target[i]=1;mask[i]=1
            continue
        assertions=[]
        for sentence in sentences:
            for match in re.finditer(r'\b(?:'+pattern+r')\b',sentence):
                prefix=sentence[:match.start()]
                if UNCERTAIN.search(sentence):
                    assertions.append(None);continue
                prefix=re.sub(r'\bno (?:significant )?(?:interval )?change\b','stable',prefix)
                # A negation covers coordinated findings until an explicit contrast.
                negative=bool(NEG.search(prefix)) or bool(re.search(r'\b(?:is|are) (?:absent|not seen|resolved)\b',sentence[match.end():]))
                assertions.append(0.0 if negative else 1.0)
        if 1.0 in assertions:target[i]=1;mask[i]=1
        elif 0.0 in assertions:target[i]=0;mask[i]=1
        elif assertions:mask[i]=0
    return target,mask
