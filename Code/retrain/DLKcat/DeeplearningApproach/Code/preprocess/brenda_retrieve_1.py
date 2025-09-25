#!/usr/bin/python
import os
import time
import random
import hashlib
from tqdm import tqdm
from zeep import Client

# INPUTS:
output_path = '../../Data/database/brenda_ec'
last_field = ''
last_EC = ''
email = '1055285901@qq.com'
password = "LBXSQJLRTZ1124"

# Construct BRENDA client:
wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
client = Client(wsdl)
password = hashlib.sha256(password.encode('utf-8')).hexdigest()
credentials = (email,password)

# Function to extract all BRENDA data from a specific field.
def extract_field(field, last):
    ECnumbers = client.service.getEcNumbersFromEcNumber(*credentials)
    print(f"Retrieved {len(ECnumbers)} EC numbers.")
    start = False
    for ECnumber in ECnumbers:
        if not start and (ECnumber == last or last == ''):
            start = True
        if start:
            query = (email, password, f"ecNumber*{ECnumber}", "turnoverNumber*", "turnoverNumberMaximum*", "substrate*", "commentary*", "organism*", "ligandStructureId*", "literature*")
            success = 0
            while success < 10:
                try:
                    file_name = os.path.join(output_path, f'EC{ECnumber}_{field}.txt')
                    data = getattr(client.service.getTurnoverNumber(*query))

                    if data:
                        with open(file_name, 'w') as fid:
                            fid.write(data.decode('ascii', 'ignore'))
                        # print(f"Successfully saved {file_name}")
                        break

                except Exception as e:
                    print(f"Error encountered: {e}. Retrying...")
                    time.sleep(random.random() * 3)
                    success += 1
# Main script
prev_path = os.getcwd()
os.chdir(output_path)

fields = ['KCAT']
start = False
for field in tqdm(fields):
    if not start and (field == last_field or last_field == ''):
        start = True

    if start:
        extract_field(field, last_EC)

os.chdir(prev_path)