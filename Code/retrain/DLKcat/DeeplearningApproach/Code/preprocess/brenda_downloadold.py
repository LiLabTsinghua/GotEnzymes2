#!/usr/bin/python
################################################################################
# createECfiles
# Reads all data in kinetic_data and creates all EC files.
#
# Benjamin Sanchez. Last edited: 2018-04-10
################################################################################

# Updated by:
# Author: LE YUAN
# This code should be run under the Python 2.7 environment

import os

# INPUTS:
# 1) Path in which all BRENDA queries are (from script retrieveBRENDA.py):
input_path = '../../Data/database/brenda_ec'
# 2) Path in which you wish to store all EC files:
output_path = '../../Data/database/Kcat_brenda'

################################################################################

def read_brenda_files(input_path):
    """Reads all BRENDA file names and sorts them."""
    dir_files = sorted(os.listdir(input_path))
    return dir_files

def process_file(input_path, filename):
    """Reads and processes the content of a BRENDA file."""
    with open(os.path.join(input_path, filename), 'r') as file:
        data = file.read()
    return data

def parse_data(ec_number, var_name, data):
    """Parses data from the BRENDA file according to the variable name."""
    ec_table = []
    
    if var_name == 'KM':
        variable = '#kmValue*'
    elif var_name == 'MW':
        variable = '#molecularWeight*'
    elif var_name == 'PATH':
        variable = '#pathway*'
    elif var_name == 'SEQ':
        variable = '#sequence*'
    elif var_name == 'SA':
        variable = '#specificActivity*'
    elif var_name == 'KCAT':
        variable = '#turnoverNumber*'
    
    options = data.split(variable)
    for k in options:
        value_pos = k.find('#')
        if value_pos != -1:
            k_value = k[:value_pos]
            k_split = k.split('#substrate*')
            k_substrate = k_split[1][:k_split[1].find('#')] if len(k_split) > 1 else '*'
            k_comment = k.split('#commentary')[1][:k.find('#')] if '#commentary' in k else '*'
            k_org = k.split('#organism*')[1][:k.find('#')] if '#organism*' in k else '*'
            
            ec_table.append(f"{var_name}\t{k_org}\t{k_value}\t{k_substrate}\t{k_comment}\n")
    
    return ec_table

def write_ec_file(output_path, ec_number, ec_table):
    """Writes the EC table data to a file."""
    output_file = os.path.join(output_path, f"{ec_number}.txt")
    with open(output_file, 'w') as file:
        file.writelines(ec_table)
    # print(f'Successfully constructed {ec_number} file.')

def main(input_path, output_path):
    dir_files = read_brenda_files(input_path)
    previous_ec = ''
    ec_table = []

    for filename in dir_files:
        sep_pos = filename.find('_')
        ec_number = filename[:sep_pos]
        var_name = filename[sep_pos+1:-4]

        data = process_file(input_path, filename)

        if ec_number != previous_ec and previous_ec:
            write_ec_file(output_path, previous_ec, ec_table)
            ec_table = []

        ec_table.extend(parse_data(ec_number, var_name, data))
        previous_ec = ec_number

    # Write the last EC file
    if ec_table:
        write_ec_file(output_path, previous_ec, ec_table)
    print(f'Successfully constructed')

if __name__ == "__main__":
    main(input_path, output_path)

################################################################################
