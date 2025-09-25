import pandas as pd
from zeep import Client
import hashlib
import time

# 示例DataFrame，实际使用时请替换为您的DataFrame
data = {
    'ecNumber': ['1.1.1.1', '2.7.11.1'],
    'organism': ['Mus musculus', 'Escherichia coli']
}
df = pd.DataFrame(data)

# 设置BRENDA API访问信息
wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
email = "1055285901@qq.com"  # 您的BRENDA账户电子邮件
password = hashlib.sha256("LBXSQJLRTZ1124".encode("utf-8")).hexdigest()  # 您的密码
client = Client(wsdl)

# 从BRENDA API获取序列的函数，增强鲁棒性
def get_sequence(ec_number, organism, attempts=3, delay=2):
    parameters = (
        email,
        password,
        f"ecNumber*{ec_number}",
        "sequence*",
        "noOfAminoAcids*",
        "firstAccessionCode*",
        "source*",
        "id*",
        f"organism*{organism}"
    )
    for attempt in range(attempts):
        try:
            # print(parameters)
            results = client.service.getSequence(*parameters)
            # print(results)
            if results:
                # 选择第一条结果中的序列，可以根据需要修改选择逻辑
                sequence = results[0]['sequence']
                return sequence
            else:
                print(f"No result found for EC {ec_number} and organism {organism} on attempt {attempt + 1}")
        except Exception as e:
            print(f"Failed to fetch sequence for EC {ec_number} and organism {organism} on attempt {attempt + 1}: {e}")
        time.sleep(delay)
    return None

# 在DataFrame中为序列创建一个新列
df['Sequence'] = df.apply(lambda row: get_sequence(row['ecNumber'], row['organism']), axis=1)

# 查看结果
print(df)
