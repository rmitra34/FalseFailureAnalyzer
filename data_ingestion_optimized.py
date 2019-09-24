"""Get data from quincy server using the csv file,
clean the data, merge all logs in one file and
save in corresponding buckets."""

import os
import csv
import sys
import time
from Preprocessing import Preproc


def get_data_from_csv(src_csv, target_dir):
    """Downloads the dr_data using the src_csv on target_dir
    Parameters: src_csv, destination dir
   """
    file = open(src_csv)
    row_count = len(file.readlines())
    print(row_count)

    preprocess = Preproc()

    with open(src_csv, encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            log_path = row['logpath']
            exit_code = row['reg_exitcode']
            test_exec_id = row['test_exec_id']
            debug_tag = row['debug_tag']

            if log_path and exit_code == 'FAIL':

                bucket = preprocess.get_class(debug_tag)

                if bucket:
                    destination = os.path.join(os.getcwd(), target_dir,
                                               bucket)

                    if not os.path.exists(destination):
                        os.makedirs(destination)

                    destination = os.path.join(destination, test_exec_id)

                    # Check if path exists on the server
                    print(log_path, destination, i, row_count)
                    if os.path.exists(log_path):
                        preprocess.get_logs(log_path, destination)


# print(read_sample('/Users/patila/Desktop/folder/67739634'))
def main():
    """Driver code
    Parameters :csv_file_path & destination_folder_name.
   """
    if len(sys.argv) > 1 and len(sys.argv) == 3:
        src_csv = sys.argv[1]
        target_dir = sys.argv[2]
        s_time = time.time()
        get_data_from_csv(src_csv, target_dir)
        print('Done in ', str((time.time()-s_time)//60)+' minutes')
    else:
        print('Takes 2 arguments: csv_file_path & destination_folder_name.')

    print('Finished')


if __name__ == '__main__':
    main()
