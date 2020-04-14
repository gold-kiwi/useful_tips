import os
import sys

def get_file_list(dir_path,ext='ALLFILE',with_dir_path=True):
    file_name_list = os.listdir(dir_path)
    true_list = []
    for file_name in file_name_list:
        if os.path.isfile(os.path.join(dir_path,file_name)):
            if not ext == 'ALLFILE':
                if os.path.splitext(file_name)[1] == ext:
                    if with_dir_path:
                        true_list.append(os.path.join(dir_path,file_name))
                    else:
                        true_list.append(file_name)
            else:
                if with_dir_path:
                    true_list.append(os.path.join(dir_path,file_name))
                else:
                    true_list.append(file_name)

    return true_list

def get_directory_list(dir_path,with_dir_path=True):
    dir_name_list = os.listdir(dir_path)
    true_list = []
    for dir_name in dir_name_list:
        if os.path.isdir(os.path.join(dir_path,dir_name)):
            if with_dir_path:
                true_list.append(os.path.join(dir_path,dir_name))
            else:
                true_list.append(dir_name)

    return true_list

