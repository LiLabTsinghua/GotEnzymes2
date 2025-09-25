

# with open(r"D:/Code/BIO/brenda-main/files/TN.txt", "r",encoding='utf-8',newline="")as f:
# # with open ("test.txt","r") as f:
#     f_list = f.readlines()  
#     print(f_list[0].strip().split('\t')) 
#     print(f_list[1].strip().split('\t')) 

keyword = "TN" # total 81451
#keyword = "KM"
#keyword = "KI"
import csv
import re

'''
#初实验
with open("D:/Code/BIO/brenda-main/files/" + keyword + ".txt","r")as f:
    lines = f.readlines()
    with open(keyword + ".tsv","w",encoding="gbk",newline="") as outfile:
        writer =csv.writer(outfile,delimiter='\t')
        # headers=['EC-number','Organism','Substrate','Value','Commentary','Reference','Protein-ID']
        # data=f_list.strip().split('\t')
        # print(data)
        # writer.writerow(headers)
        for line in lines:
            # outfile.write(line.replace(',',' ').replace('\t',','))
            outfile.write(line)
        # f_list.strip().split('\t')
        # print(f_list[0].strip().split('\t')) 
'''

'''
下面为对接brenda_kcat_preprocess
'''
def is_float(string):#先定义一个检测是否为float的函数
    pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'
    if re.match(pattern, string):
        return True
    else:
        return False


# with open("D:/Code/BIO/brenda-main/files/" + keyword + ".txt","r")as f:
with open("../../Data/database/" + keyword + ".txt","r")as f:
    lines = f.readlines()
    with open(keyword + ".tsv","w",encoding="gbk",newline="") as outfile:
        tsv_writer =csv.writer(outfile,delimiter='\t')
        tsv_writer.writerow(["EntryID", "Type", "ECNumber", "Substrate", 'EnzymeType', "Organism","Value", "Unit"])
        #total is 8
        i=0
        j=0
        k=0
        t=0
        # t1=0
        nb=0
        for line in lines[1:]:
            #tsv_writer.writerow([i,'kcat',line])
            data = line.strip().split('\t')
            desc = data[4]#should be describe--close to commentary
            if(is_float(data[3])):
                value = data[3]
                if float(value) > 0 :  # Kcat value exist some weird values eg.-999.
                    if 'mutant' in desc or 'mutated' in desc:
                        # print(desc)
                        mutant = re.findall('[A-Z]\d+[A-Z]', desc)  # re is of great use
                        # print(mutant)
                        t+=1
                        if len(mutant) >=1 :#有标注变异
                            enzymeType = '/'.join(mutant)
                            # t1+=1
                        else :
                            nb+=1
                            continue#没有定义哪一个位变异了
                    else :
                        enzymeType = 'wildtype'
                    i+=1
                    if(keyword=='TN'):
                        tsv_writer.writerow([i, 'kcat', data[0], data[2], enzymeType, data[1], data[3], 's^(-1)'])
                    elif(keyword=='KM'):
                        tsv_writer.writerow([i, 'KM', data[0], data[2], enzymeType, data[1], data[3], '???'])
                    else:          #KI
                        tsv_writer.writerow([i, 'KI', data[0], data[2], enzymeType, data[1], data[3], '???'])
                else:
                    k+=1
                    # print(value)
            # else:#若删除2-8类的数据
            #     j+=1
            #     # print(data[3])
            #     # print(i)
            else:#计数后删除
                j+=1
                # value=data[3][data[3].find('-')+1:]
                # if 'mutant' in desc or 'mutated' in desc:
                #     # print(desc)
                #     mutant = re.findall('[A-Z]\d+[A-Z]', desc)  # re is of great use
                #     # print(mutant)
                #     t+=1
                #     if len(mutant) >=1 :#多变异
                #         enzymeType = '/'.join(mutant)
                #         t1+=1
                #     else :
                #         nb+=1
                #         continue
                # else :
                #     enzymeType = 'wildtype'
                # i+=1
                # if(keyword=='TN'):
                #     tsv_writer.writerow([i, 'kcat', data[0], data[2], enzymeType, data[1], data[3], 's^(-1)'])
                # elif(keyword=='KM'):
                #     tsv_writer.writerow([i, 'KM', data[0], data[2], enzymeType, data[1], data[3], '???'])
                # else:          #KI
                #     tsv_writer.writerow([i, 'KI', data[0], data[2], enzymeType, data[1], data[3], '???'])
            # if(i==1000):
            #     print(data)
            #     #经实验，无需添加value与0的代码块，即这里没有value小于等于0的情况（可能未必，程序把-当成到了，没当成负号
            #     # print(value)
            #     # if(is_float(value)!=1):
            #     #     t+=1
            #     #     print(i)
            #     #     print(data)
            #     #测试，多了很多数据，是因为写文件模式为'a'
print("因Kcat有范围删除的数据个数：%d" % j)#total 640
print("因Kcat小于等于0删除的数据个数：%d" % k)#total 1275
print("变异数据个数：%d" % t)#27388
# print("变异个数：%d" % t1)#26154
print("因变异没标注哪一位而被删除的数据个数：%d" % nb)#1234
print("剩余的数据个数：%d" % (i))#78302
#删除；kcat剩了78302条数据
#取最大值：kcat剩了78923条数据?
#删除；kcat剩了78302条数据
#取最大值：kcat剩了80175条数据
#78924+2*1275=81451?
#问题：i=1000,1510,1511处自动加了1
#解决：是因为100行的continue
#i=1000处,EC=1.1.1.144唯一
'''
实验
# 原始字符串包含转义字符
original_string = "这是一段包含\\n换行符和\\t制表符的文本"
print("原始字符串:\n", original_string)

# 使用replace()函数进行替换
new_string = original_string.replace('\\n', '\n').replace('\\t', '\t')
print("\n替换后的字符串:\n", new_string)
str1="1.1.1.1	Rattus norvegicus	1-butanol	0.833	isoenzyme ADH-3, pH 10.0	 <49>	"
str2=str1.replace('\t', '666')
print(str1)
print(str2)
'''