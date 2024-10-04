import sys
import zipfile
import pandas as pd
import numpy as np
import os
import re
import csv

def make_zipfile(output_filename, source_dir) -> bool:
    """
    This is a helper function that generates a zip file after the labels have been transformed
    It reads the target directory
    """
    try:
        relroot = os.path.abspath(os.path.join(source_dir, os.pardir))
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zip:
            for root, dirs, files in os.walk(source_dir):
                zip.write(root, os.path.relpath(root, relroot))
                for file in files:
                    filename = os.path.join(root, file)
                    if os.path.isfile(filename): # regular files only
                        arcname = os.path.join(os.path.relpath(root, relroot), file)
                        zip.write(filename, arcname)
        return True
    except:
        return False

def transform_labels(zip_path: str, target_dir: str) -> None:
    """
    This program reassigns statement-level labels for cpp program files
    It reads a compressed directory of csv files delimited by ` and target dir
    Each csv file represents a cpp program file.
    The first column represents the program syntax, subsequent columns represent encodings based on SLDEEP
    The last column represents a label, where 0 represents no error and 1 reperesents an error. 
    This program reassigns the encountered errors as follows: 
    0: no error, 1:unclassified errors, 2: branch errors, 3: loop errors 4: declaration errors
    It writes the files to a 'validated' folder in the target dir and compresses it when done transforming all CSVs
    """
    declaration_error_regex = r'((?<=std\:\:)?(?:\w{3,10})\s+[a-zA-Z_]{1,}[0-9]{0,}\s+?\=\s+?[a-zA-Z0-9\,\.\"\']{1,}\;)'
    branch_error_regex = r'(?:if\s{0,}\()|(?:\?\s{0,}.+\:\s{0,}.+\;$)'
    loop_error_regex = r'(?:while\s{0,}\()|(?:for\s{0,}\()'
    file_name_regex = r'\d+\.cpp\.csv'

    target_folder = os.path.join(target_dir, 'validated')
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files in the ZIP archive
        zip_file_list = zip_ref.namelist()
    
        # Filter CSV files
        valid_csv_file_regex = r'^(?!<\._)(?:[a-zA-Z]{0,}[\\\/])?\d+\.cpp\.csv'
        csv_files = [file for file in zip_file_list if re.match(valid_csv_file_regex, file)]
        
        for file in csv_files:
            
            file_name = re.search(file_name_regex, file).group() 
            with zip_ref.open(file) as csv_file:
                print(f'currently reading {file_name}')
                # try:
                program = pd.read_csv(csv_file, delimiter='`',header=0, encoding='utf-8')
                # except:
                #     print(f'Could not read {file_name}\nskipping..')
                #     continue
            for idx, LOC in program.iterrows():
                if (LOC.iloc[-1] == 1):
                    if re.search(declaration_error_regex,LOC.iloc[0]): #case declaration error
                        program.iloc[idx, -1] = 4
                        print(f'declaration error found in {file_name} at line {idx}')
                    if re.search(branch_error_regex,LOC.iloc[0]): #case branch error
                        program.iloc[idx, -1] = 2
                        print(f'branch error found in {file_name} at line {idx}')
                    if re.search(loop_error_regex,LOC.iloc[0]): #case loop error
                        program.iloc[idx, -1] = 3
                        print(f'loop error found in {file_name} at line {idx}')

            new_file_name = os.path.join(target_folder, file_name)

            program.to_csv(new_file_name, sep='`', index=False, quoting=csv.QUOTE_NONE, quotechar='')
            print(f'done reading and writing {file_name}')
            
    
    
    zip_dir = os.path.join(target_dir, "newdata.zip")
    ZIP_STATUS = make_zipfile(zip_dir, target_folder)
    if not ZIP_STATUS:
        print('Could not make zip file, zip the folder manually')
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(transform_labels.__doc__)
        print(f'Usage: python <zip_file_path> <target_dir>')
    else:
        # try:
        transform_labels(sys.argv[1], sys.argv[2])
        # except:
        #     print('An error occured printing the files')
