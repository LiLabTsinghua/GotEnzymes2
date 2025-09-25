############################################################################################################
# get EC numbers from Brenda
# fetch data from brenda
output_file = 'kcat_km_values.csv'

import hashlib
import pandas as pd
from zeep import Client
from zeep.exceptions import Fault, TransportError
from requests.exceptions import ConnectionError, ChunkedEncodingError
import time
import os
from tqdm import tqdm
wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
email = "1055285901@qq.com"
password = hashlib.sha256("LBXSQJLRTZ1124".encode("utf-8")).hexdigest()
# 创建SOAP客户端
client = Client(wsdl)
# 查询所有 EC numbers
parameters_ec = (email, password)
ECnumbers = client.service.getEcNumbersFromEcNumber(*parameters_ec)
print(f"Retrieved {len(ECnumbers)} EC numbers.")

# 初始化存储数据的列表
data = []

# 加载已存在的数据（如果有）
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    processed_ec_numbers = set(df_existing['EC_number'].unique())
    data = df_existing.to_dict('records')
else:
    processed_ec_numbers = set()

def fetch_data(client, email, password, ecNumber, service_method, max_retries=5):
    for attempt in range(max_retries):
        try:
            if service_method == "kcat":
                parameters = (email, password, f"ecNumber*{ecNumber}", "turnoverNumber*", "turnoverNumberMaximum*", "substrate*", "commentary*", "organism*", "ligandStructureId*", "literature*")
                return client.service.getTurnoverNumber(*parameters)
            elif service_method == "km":
                parameters = (email, password, f"ecNumber*{ecNumber}", "kmValue*", "kmValueMaximum*", "substrate*", "commentary*", "organism*", "ligandStructureId*", "literature*")
                return client.service.getKmValue(*parameters)
        except (ConnectionError, ChunkedEncodingError, Fault, TransportError) as e:
            print(f"Attempt {attempt + 1} failed for EC {ecNumber}: {e}")
            time.sleep(2 ** attempt)  # 指数回退重试
    return None

# 遍历 EC numbers 列表并提取数据
for ecNumber in tqdm(ECnumbers):
    if ecNumber in processed_ec_numbers:
        print(f"EC number {ecNumber} 已处理，跳过...")
        continue
    kcat_results = fetch_data(client, email, password, ecNumber, "kcat")
    km_results = fetch_data(client, email, password, ecNumber, "km")

    # 处理并存储 kcat 数据
    if kcat_results:
        for result in kcat_results:
            data.append({
                "EC_number": ecNumber,
                "Type": "kcat",
                "Value": getattr(result, 'turnoverNumber', ''),
                "Maximum": getattr(result, 'turnoverNumberMaximum', ''),
                "Substrate": getattr(result, 'substrate', ''),
                "Commentary": getattr(result, 'commentary', ''),
                "Organism": getattr(result, 'organism', ''),
                "LigandStructureId": getattr(result, 'ligandStructureId', ''),
                "Literature": getattr(result, 'literature', '')
            })
    # 处理并存储 km 数据
    if km_results:
        for result in km_results:
            data.append({
                "EC_number": ecNumber,
                "Type": "km",
                "Value": getattr(result, 'kmValue', ''),
                "Maximum": getattr(result, 'kmValueMaximum', ''),
                "Substrate": getattr(result, 'substrate', ''),
                "Commentary": getattr(result, 'commentary', ''),
                "Organism": getattr(result, 'organism', ''),
                "LigandStructureId": getattr(result, 'ligandStructureId', ''),
                "Literature": getattr(result, 'literature', '')
            })

    # 每次处理完一个EC number后，保存数据到文件
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"EC number {ecNumber} 的数据已保存。")

print("所有数据提取并保存完成")