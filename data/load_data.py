from urllib import request
import os
import pandas as pd

# Download data module
module_url = f"https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/dont_patronize_me.py"
module_name = module_url.split('/')[-1]
print(f'Fetching {module_url}')
#with open("file_1.txt") as f1, open("file_2.txt") as f2
with request.urlopen(module_url) as f, open('data/'+module_name,'w') as outf:
  a = f.read()
  outf.write(a.decode('utf-8'))


# helper function to save predictions to an output file
def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')
			
from dont_patronize_me import DontPatronizeMe

dpm = DontPatronizeMe('data', '.')

# Load raw data
tsv_url = "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/refs/heads/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv"
with request.urlopen(tsv_url) as f, open("data/dontpatronizeme_pcl.tsv", "w") as outf:
    outf.write(f.read().decode('utf-8'))

dpm.load_task1()

# --- DEV SET ---
base = "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/practice%20splits/"
dev_ids = pd.read_csv(base + "dev_semeval_parids-labels.csv")
data = dpm.train_task1_df

rows = []
for idx in range(len(dev_ids)):
    parid = str(dev_ids.par_id[idx])
    keyword = data.loc[data.par_id == parid].keyword.values[0]
    text = data.loc[data.par_id == parid].text.values[0]
    label = data.loc[data.par_id == parid].label.values[0]
    orig_label = data.loc[data.par_id == parid].orig_label.values[0]
    rows.append({
        'par_id': parid,
        'keyword': keyword,
        'paragraph': text,
        'label': label,
        'original_label': orig_label
    })

dev_df = pd.DataFrame(rows)
os.makedirs('data', exist_ok=True)
dev_df.to_csv('data/dev.csv', index=False)
print(f"Dev set saved: {len(dev_df)} rows → data/dev.csv")

# --- TEST SET ---
test_tsv_url = "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/refs/heads/master/semeval-2022/TEST/task4_test.tsv"
with request.urlopen(test_tsv_url) as f, open("data/task4_test.tsv", "w") as outf:
    outf.write(f.read().decode('utf-8'))

dpm_test = DontPatronizeMe('.', 'data/task4_test.tsv')
dpm_test.load_test()
test_df = dpm_test.test_set_df[['par_id', 'keyword', 'text']].rename(columns={'text': 'paragraph'})
test_df.to_csv('data/test.csv', index=False)
print(f"Test set saved: {len(test_df)} rows → data/test.csv")