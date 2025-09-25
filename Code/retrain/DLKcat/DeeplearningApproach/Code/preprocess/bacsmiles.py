# 不太好用
import pandas as pd
import requests
import json
import concurrent.futures
from tqdm import tqdm

df = pd.read_csv('kcat_data.csv')

df = df.head(5)

not_found = []  # 记录未找到的化合物列表
err_cnt = 0  # 错误计数器

session = requests.Session()  # 使用Session以提高请求效率

def get_smiles(substrate, attempts=3):
    global err_cnt
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{substrate}/property/CanonicalSMILES/JSON"
    for attempt in range(attempts):
        try:
            response = session.get(url)
            if response.status_code == 200:
                smiles_data = response.json()
                return smiles_data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            else:
                # 如果状态码不是200，打印状态码和错误信息
                print(f"Attempt {attempt+1}: Received status code {response.status_code} for {substrate}")
                if attempt == attempts - 1:
                    not_found.append(substrate)
                    err_cnt += 1
        except requests.exceptions.RequestException as e:
            # 包括所有可能的请求错误（如连接问题、超时等）
            print(f"Request exception on attempt {attempt+1} for {substrate}: {e}")
            if attempt == attempts - 1:
                not_found.append(substrate)
                err_cnt += 1
        except Exception as e:
            # 其他未预见的错误
            print(f"Other error on attempt {attempt+1} for {substrate}: {e}")
            if attempt == attempts - 1:
                not_found.append(substrate)
                err_cnt += 1
    return None

unique_substrates = set(df['Substrate'])

# 使用并发请求以提高速度
def fetch_smiles(substrate):
    return substrate, get_smiles(substrate)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # 可以调整max_workers以优化性能
    results = list(tqdm(executor.map(fetch_smiles, unique_substrates), total=len(unique_substrates)))

substrate_smiles_dict = dict(results)
df['Smiles'] = df['Substrate'].map(substrate_smiles_dict)

print("Number of substrates not found:", len(not_found))
print("List of not found substrates:", not_found)
print("Number of errors:", err_cnt)

# Optional: save not found substrates to a file
with open('not_found_substrates.json', 'w') as f:
    json.dump(not_found, f)

# save to json
with open('substrate_smiles.json', 'w') as f:
    json.dump(substrate_smiles_dict, f)
