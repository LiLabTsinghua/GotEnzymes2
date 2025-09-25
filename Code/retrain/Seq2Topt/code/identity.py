import pandas as pd
from tqdm import tqdm
from Bio.Emboss.Applications import NeedleCommandline
from multiprocessing import Pool, freeze_support
import os

def calculate_identity(fasta_file_1, fasta_file_2):
    needle_cline = NeedleCommandline(asequence=fasta_file_1, bsequence=fasta_file_2,
                                     gapopen=10, gapextend=0.5, filter=True)
    out = needle_cline()[0]
    out = out[out.find("Identity"):]
    out = out[:out.find("\n")]
    identity = float(out[out.find("(")+1:out.find(")")-1].replace(" ", "")) / 100
    return round(identity, 3)

def calculate_max_identity(args):
    test_fasta_path, train_fasta_paths = args
    max_identity = -1
    for train_fasta_path in train_fasta_paths:
        try:
            identity = calculate_identity(test_fasta_path, train_fasta_path)
            if identity > max_identity:
                max_identity = identity
        except Exception as e:
            print(f"Error calculating identity between {test_fasta_path} and {train_fasta_path}: {e}")
    
    return max_identity

def write_fasta_file(sequence, output_path):
    with open(output_path, 'w') as f:
        f.write(f">sequence\n{sequence}\n")

def generate_fasta_files(df, prefix, output_dir):
    unique_sequences = df.drop_duplicates(subset=['sequence'])
    unique_sequences.reset_index(drop=True, inplace=True)
    sequence_to_idx = {}
    for idx, row in unique_sequences.iterrows():
        fasta_file = os.path.join(output_dir, f'{prefix}_{idx}.fasta')
        write_fasta_file(row['sequence'], fasta_file)
        sequence_to_idx[row['sequence']] = fasta_file
    return sequence_to_idx

def main():
    # 加载数据
        # data_df = pd.read_csv(f'/home/wuke/project/bio_deeplearning/zzz_benchmark/data/EITLEM_{km}.csv')
        # data_df = pd.read_csv(f'../data/Topt/train_os_cv0.csv')
        # output_dir = f'./topt_identity'
        # output_file = f'Seq2Topt_topt_identity.csv'
        # EITLEM_kcat_test_df = pd.read_csv(f'../data/Topt/test_cv0.csv')
        # EITLEM_kcat_train_df = pd.read_csv(f'../data/Topt/train_os_cv0.csv')
        output_dir = f'./tm_identity_new'
        output_file = f'../data/Tm/Seq2Topt_tm_identity_merged.csv'
        EITLEM_kcat_test_df = pd.read_csv(f'../data/Tm/Tm_new_Test_cv0.csv')
        EITLEM_kcat_train_df = pd.read_csv(f'../data/Tm/Tm_new_Train_cv0.csv')
        EITLEM_kcat_test_df['Test'] = 1
        EITLEM_kcat_train_df['Test'] = 0
        data_df = pd.concat([EITLEM_kcat_test_df, EITLEM_kcat_train_df], ignore_index=True)
        # 创建 lbx_fasta_data 目录
        os.makedirs(output_dir, exist_ok=True)

        # 提取唯一序列并生成 FASTA 文件
        test_sequence_to_fasta = generate_fasta_files(EITLEM_kcat_test_df, 'test', output_dir)
        train_sequence_to_fasta = generate_fasta_files(EITLEM_kcat_train_df, 'train', output_dir)

        # 准备参数列表
        train_fasta_paths = list(train_sequence_to_fasta.values())
        args_list = [(test_fasta_path, train_fasta_paths) for test_fasta_path in test_sequence_to_fasta.values()]

        # 计算最大相似度
        with Pool(processes=80) as pool:  # 根据你的 CPU 核心数设置进程数
            max_identities = list(tqdm(
                pool.imap(calculate_max_identity, args_list),
                total=len(args_list)
            ))

        # 映射结果到原始 DataFrame
        for i, (sequence, fasta_path) in enumerate(test_sequence_to_fasta.items()):
            original_indices = EITLEM_kcat_test_df[EITLEM_kcat_test_df['sequence'] == sequence].index.tolist()
            for original_idx in original_indices:
                data_df.loc[original_idx, 'identity'] = max_identities[i]

        # 保存更新后的 DataFrame 回 CSV 文件
        data_df.to_csv(output_file, index=False)
        print(f"Processing completed and updated data saved to {output_file}.")
        data_df = data_df[data_df['Test']==1]
        data0 = pd.read_csv('test_cv0_tm_merged.csv')
        data_df['prediction'] = data0.iloc[:, -1].values
        data_df.to_csv('Seq2Topt_tm_identity_new.csv', index=False)

if __name__ == '__main__':
    main()
