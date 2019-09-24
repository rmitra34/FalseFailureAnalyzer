"""Get data from quincy server using the csv file,
clean the data, merge all logs in one file and
save in corresponding buckets."""

import os
from stat import S_ISDIR
import csv
import shutil
import re
import sys

import paramiko

paramiko.util.log_to_file('/tmp/paramiko.log')

HOST = "ttqc-shell03"
PORT = 22
TRANSPORT = paramiko.Transport((HOST, PORT))
USERNAME = "testtool"
PASSWORD = "B3k1nD+"
TRANSPORT.connect(username=USERNAME, password=PASSWORD)
SFTP = paramiko.SFTPClient.from_transport(TRANSPORT)

# buckets = ['Hardware', 'Other', 'Script', 'Software', 'Tools']

BUCKETS = dict({'Hardware': ['HW'], 'Software': ['SW', 'TOXIC-PR', 'TOXIC-BUILD'],
                'Script': ['SCRIPT'], 'Tools': ['TOOLS']})

BUCKETS['Other'] = ['FALSE-FAILURE-ANALYZED', 'FALSE-FAILURE-RESOLVED', 'JBATCH-UPDATE',
                    'AUTO-UPDATE', 'PARAMS-ISSUE', 'OTHER', 'matchUp']

RE_LIST = [
    r'(Jan?|Feb?|Mar?|Apr?|May|Jun?|Jul?|Aug?|Sep?|Oct?|Nov?|Dec?)\s+\d{1,2}\s+',
    r'\d{2}:\d{2}:\d{2}\s+',
    r'[\[].*?[\]]',
    r'<\/?data>',
    r'<\/?value>',
    r'<\/?valueType>',
    r'<\/?name>',
    r'<\/?valueUnit>',
    r'<\/?object-value>',
    r'<\/?object-value-type>',
    r'<\/?cli>',
    r'<[0-9]{4,5}>',
    r'<\/?DATA>'
]


# Remove all extraneous tags from all of the log files and absolute path of script
def trim_logs(filepath):
    """Performs intial cleaning of logs as per RE_LIST.
    Parameters: file path
    Returns: edited text from log
   """
    # Remove absolute path found in beginning of script log.
    script_marker = 'script-exec/'
    text = open(filepath, errors='replace').read()
    text = re.sub(r'[\[]ERROR[\]]\s+', 'ERROR ', text)  # Flags any ERROR tag in the log.

    for i in RE_LIST:
        text = re.sub(i, '', text)
    new_line = text.find(script_marker)

    text = text[new_line + len(script_marker):]
    return text


def parse_xml(*filepath):
    """Remove xml tags and merger output_xml and .log files.
    Parameters: file path
    Returns: unique text from log
   """
    text = re.sub(r'<[^<]+>', "", open(filepath[0], errors='ignore').read())
    # Rid of all tags from XML file
    text = re.sub(r'[\[].*[\]]', '', text)
    lines = text.split("\n")
    unique_lines = set(lines)
    # Create list of all unique lines found in XML file
    unique_lines = list(unique_lines)
    for i, u_line in enumerate(unique_lines):
        if u_line:
            unique_lines[i] = raw_string_xml(unique_lines[i])
    unique_lines = set(unique_lines)

    if len(filepath) > 1:
        with open(filepath[1]) as file:
            line = file.readline()
            while line:
                # Get rid of all extraneous characteristics in string
                davo = raw_string_log(line)
                # Append line to text to be added to .txt file if not in unique_lines
                if davo not in unique_lines:
                    text += line + '\n'
                line = file.readline()
    return text


# Get rid of extraneous characters in lines
def raw_string_xml(text):
    """Remove new lines and tabs from text.
    Parameters: string
    Returns: string
   """
    text = text.lower().lstrip()
    text = text.rstrip('\n')
    text = text.rstrip('\r')
    return text


def raw_string_log(line):  # Remove all tags from file
    """Remove regex matching lines from text.
    Parameters: string
    Returns: string
   """
    davo = re.sub(r'[\[].*[\]]', '', line)
    davo = davo.rstrip('\n')
    davo = davo.rstrip('\r')
    davo = davo.replace('<', '&lt')
    davo = davo.replace('>', '&gt')
    davo = davo.lower().lstrip()
    return davo


def sftp_walk(remote_path):
    """Walks the remote path.
    Parameters: dir path on remote
    Returns: path, sub-dir, list of files
   """
    path = remote_path
    files = []
    folders = []
    for file in SFTP.listdir_attr(remote_path):
        if S_ISDIR(file.st_mode):
            folders.append(file.filename)
        else:
            files.append(file.filename)
    if files:
        yield path, files
    for folder in folders:
        new_path = os.path.join(remote_path, folder)
        for n_file in sftp_walk(new_path):
            yield n_file


def download_file(source, destination):
    """download the source file in destination folder.
    Parameters: source path, destination path
    Returns: string: Found, Not Found
   """
    try:
        for path, files in sftp_walk(source):
            # if TOBY
            if any('output.xml' in item for item in files):
                for item in files:
                    if (item.endswith('.log') and item != 'stdout.log')\
                            or item.endswith('output.xml'):
                        if not os.path.exists(destination):
                            os.makedirs(destination)

                        SFTP.get(os.path.join(os.path.join(path, item)), destination + item)
                        # print(destination, item)

                return 'Found'
            # if JT
            if any('.pl.log' in item for item in files):

                for item in files:
                    if item.endswith('.pl.log') or item.endswith('.expect'):
                        if not os.path.exists(destination):
                            os.makedirs(destination)

                        SFTP.get(os.path.join(os.path.join(path, item)), destination + item)
                        # print(destination, item)

                return 'Found'

        return 'Not Found'

    except IOError as err:
        print('File Not Found ' + source + ' ' + str(err))
        return 'Not Found'


def clean_log(file_path):
    """Remove any duplicacy from logs
    Parameters: dir path
   """
    lines_seen = set()  # holds lines already seen
    text = trim_logs(file_path).split('\n')

    outfile = open(file_path, "w", errors='ignore')

    for line in text:
        # Removes duplicate lines
        if line not in lines_seen and len(line) < 400:
            outfile.write(line + '\n')
            lines_seen.add(line)
    outfile.close()


def combine_log(logs_path):
    """Combines all logs at path into one text file.
    Parameters: dir path
   """
    print(logs_path)
    for log_path, _, log_names in os.walk(logs_path):
        # print(logs_path,_,log_names)
        with open(logs_path[:-1] + '.txt', 'wb') as big_file:
            for log_name in log_names:
                in_file = os.path.join(log_path, log_name)

                # Removes some tags here while combining the .log and .xml
                if '.xml' in log_name:
                    # print(log_path, log_name, len(log_names))
                    expect_log = log_name.replace('_output.xml', '.log')
                    if expect_log in log_names:
                        content = parse_xml(in_file, os.path.join(log_path, expect_log))
                        log_names.remove(log_name)
                        log_names.remove(expect_log)
                    else:
                        content = parse_xml(in_file)

                    big_file.write(content.encode('utf-8'))
                else:
                    if log_name.replace('.log', '_output.xml') not in log_names:
                        with open(in_file, 'rb') as log:
                            shutil.copyfileobj(log, big_file)

            clean_log(logs_path[:-1] + '.txt')
            shutil.rmtree(logs_path[:-1])


def get_data_from_csv(src_csv='dr_data2.csv', target_dir='clean_data'):
    """Downloads the dr_data using the src_csv on target_dir
    Parameters: src_csv, destination dir
   """
    with open(src_csv, encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            log_path = row['logpath']
            exit_code = row['reg_exitcode']

            if 'prod' in log_path and exit_code == 'FAIL':
                debug_tag = row['debug_tag']
                if debug_tag not in ('0', '-'):
                    for bucket, tags in BUCKETS.items():
                        for tag in tags:
                            if tag in debug_tag:
                                if not (tag == 'OTHER' and debug_tag == 'SCRIPT-OTHER'):
                                    tmp = log_path.split('prod')[1]
                                    tmp = list(filter(lambda x: x != '', tmp.split('/')))
                                    exec_id = tmp[-3]

                                    destination = os.path.join(os.getcwd(), target_dir,
                                                               bucket, exec_id) + '/'

                                    status = download_file(log_path, destination)
                                    row['download_status'] = status

                                    if status == 'Found':
                                        combine_log(destination)


def main():
    """Driver code
    Parameters :csv_file_path & destination_folder_name.
   """
    if len(sys.argv) > 1 and len(sys.argv) == 3:
        src_csv = sys.argv[1]
        target_dir = sys.argv[2]
        get_data_from_csv(src_csv, target_dir)
        print('Done')
    else:
        print('Takes 2 arguments: csv_file_path & destination_folder_name.')

    # get_data_from_csv()


if __name__ == '__main__':
    main()
