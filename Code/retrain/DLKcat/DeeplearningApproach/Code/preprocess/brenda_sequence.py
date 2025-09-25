#!/usr/bin/python
# coding: utf-8

# E-mail in BRENDA:
email = '1055285901@qq.com'
# Password in BRENDA:
password = "LBXSQJLRTZ1124"

from zeep import Client
import hashlib

### email,password,"ecNumber*1.1.1.1#organism*Mus musculus"

wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
password = hashlib.sha256(password.encode("utf-8")).hexdigest()
client = Client(wsdl)
parameters = (email,password,"ecNumber*1.1.1.1", "sequence*",
            "noOfAminoAcids*", "firstAccessionCode*", "source*", "id*", "organism*Mus musculus")
resultString = client.service.getSequence(*parameters)
print (resultString)
# 成了！！！！！
# 修电脑后需要开vpn 未完全解决 可能是服务器问题

# # #Construct BRENDA client:
# import string
# import hashlib
# import Zeep  ## for usage without WSDL file
# endpointURL = "https://www.brenda-enzymes.org/soap/brenda_server.php"
# client      = SOAPProxy(endpointURL)
# password    = hashlib.sha256(password).hexdigest()
# credentials = email + ',' + password


# parameters = "j.doe@example.edu,"+password+","+"ecNumber*1.1.1.1#organism*Homo sapiens"
# resultString = client.getSequence(parameters)
# parameters = credentials+","+"ecNumber*1.1.1.1#organism*Homo sapiens"
# parameters = credentials+","+"ecNumber*3.1.22.4#organism*Escherichia coli"
# parameters = credentials+","+"ecNumber*3.1.3.17#organism*Oryctolagus cuniculus"
# sequence = client.getSequence("ecNumber*1.1.1.1#organism*Mus musculus")


# parameters = credentials+","+"ecNumber*4.1.1.85#organism*Escherichia coli K-12"
# sequence = client.getSequence(parameters)
# print(sequence)