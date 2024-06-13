# coding:utf-8
import argparse
import os
from pathlib import Path
import subprocess
from subprocess import check_output
import heapq
import re

import moxing as mox


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lines", type=int, default=1024, help="log lines")
    parser.add_argument("-o", "--output", help="obs path, format s3://training/log/")
    return parser.parse_args()


def print_args(input_args):
    print("upload_tail_log.py -l %d -o %s" % (input_args.lines, input_args.output))


class AscendLogFile:
    # log type
    type_plog = 'plog'
    type_device = 'device'

    """
    ascend c76 log filename format:
    ---
    plog-${pid}_${timestamp}.log, plog-1215_20210120144937340.log
    device-${pid}_${timestamp}.log, device-1215_20210120144937340.log
    ---
    specially, we assume pid is in range of 1 - 999999
    """
    proc_log_file_name_pattern = re.compile(r'plog-\d{1,6}_\d{17}.log')
    device_log_file_name_pattern = re.compile(r'device-\d{1,6}_\d{17}.log')

    def __init__(self):
        self.log_file_name = None
        self.log_type = None

        self.log_file_created_time = None
        self.log_file_pid = None

    def set_log_file(self, log_file_name):
        """
        firstly, use this func
        valid and set log file name
        """
        valid = False
        if AscendLogFile.proc_log_file_name_pattern.match(log_file_name):
            self.log_type = AscendLogFile.type_plog
            valid = True

        if AscendLogFile.device_log_file_name_pattern.match(log_file_name):
            self.log_type = AscendLogFile.type_device
            valid = True

        if not valid:
            return False

        self.log_file_name = log_file_name

        self.log_file_created_time = self.parse_created_time()
        self.log_file_pid = self.parse_pid()

        return True

    def get_log_file(self):
        return self.log_file_name

    def __lt__(self, other):
        """
        compare pid log file by created time, latest file first
        """
        return self.get_created_time() > other.get_created_time()

    def get_created_time(self):
        return self.log_file_created_time

    def get_pid(self):
        """
        return pid in log filename
        """
        return self.log_file_pid

    def parse_created_time(self):
        arr = self.log_file_name.split('_')
        if len(arr) > 1:
            timestamp_with_ext = arr[1]
            return timestamp_with_ext.split('.')[0]

        return None

    def parse_pid(self):
        arr = self.log_file_name.split('_')
        if len(arr) > 0:
            prefix = arr[0]
            arr = prefix.split('-')
        if len(arr) > 1:
            return arr[1]

        return None


class AscendLogFilePidCollector:
    """
    collect ascend log by pid
    """

    def __init__(self):
        # [id1, id2, id3, ...]
        self.pid_arr = []
        # id -> [id_file_1, id_file_2, ...]
        self.pid_to_files_map = {}

    def add_log_file(self, ascend_log_file):
        """
        add the log file to inner heap (latest file first)
        """
        log_file_pid = ascend_log_file.get_pid()
        if log_file_pid not in self.pid_to_files_map:
            self.pid_arr.append(log_file_pid)
            self.pid_to_files_map[log_file_pid] = [ascend_log_file]
        else:
            heapq.heappush(self.pid_to_files_map[log_file_pid], ascend_log_file)

    def get_pid_arr(self):
        return self.pid_arr

    def get_log_file(self, log_file_id):
        """
        get log file by its latest created time and remove it from inner heap
        """
        if log_file_id is None or \
                log_file_id not in self.pid_to_files_map or \
                len(self.pid_to_files_map[log_file_id]) == 0:
            return None

        return heapq.heappop(self.pid_to_files_map[log_file_id])


def collect_latest_n_log(new_log_dir, n):
    """
    generate limited lines(n) log file in new_log_dir
    """
    home = str(Path.home())
    ascend_log_dir = os.path.join(home, 'ascend/log')

    if not os.path.isdir(ascend_log_dir):
        print("%s is not found" % ascend_log_dir)
        return []

    print('list %s' % ascend_log_dir)
    for f in mox.file.list_directory(ascend_log_dir, recursive=True):
        print(f)

    # ~/ascend/log/plog/plog-${pid}_${timestamp}.log
    plog_collector = AscendLogFilePidCollector()

    ascend_proc_log_dir = os.path.join(ascend_log_dir, 'plog')
    if os.path.isdir(ascend_proc_log_dir):
        for target in [f for f in os.listdir(ascend_proc_log_dir)]:
            ascend_log_file = AscendLogFile()
            if ascend_log_file.set_log_file(target):
                plog_collector.add_log_file(ascend_log_file)

        generate_limited_lines_pid_log_file(new_log_dir, ascend_proc_log_dir, 'plog', plog_collector, n)

    # ~/ascend/log/device-${id}/device-${pid}_${timestamp}.log
    device_id_pattern = re.compile(r'device-[0-7]')
    for device_id_dir in [log_dir for log_dir in os.listdir(ascend_log_dir) if device_id_pattern.match(log_dir)]:
        device_log_collector = AscendLogFilePidCollector()
        ascend_device_log_dir = os.path.join(ascend_log_dir, device_id_dir)

        for target in [f for f in os.listdir(ascend_device_log_dir)]:
            ascend_log_file = AscendLogFile()
            if ascend_log_file.set_log_file(target):
                device_log_collector.add_log_file(ascend_log_file)

        generate_limited_lines_pid_log_file(new_log_dir, ascend_device_log_dir, device_id_dir, device_log_collector, n)


def generate_limited_lines_pid_log_file(new_log_dir, log_dir, new_log_short_name, log_collector, limited_lines):
    for pid in log_collector.get_pid_arr():
        remain_lines = limited_lines
        new_log_file = os.path.join(new_log_dir, get_new_log_file_name('%s-%s' % (new_log_short_name, pid)))

        log_file = log_collector.get_log_file(pid)
        while log_file:
            log_file_path = os.path.join(log_dir, log_file.get_log_file())
            if os.path.isfile(log_file_path):
                print('collect file: %s, size: %d' % (log_file_path, os.path.getsize(log_file_path)))
            else:
                print('collect file: %s, file does not exist' % log_file_path)

            current_lines = tail_append_to_file(log_file_path, remain_lines, new_log_file)

            remain_lines = limited_lines - current_lines
            if remain_lines <= 0:
                break

            log_file = log_collector.get_log_file(pid)


def get_new_log_file_name(log_short_name):
    """
    v1: ${BATCH_JOB_ID}-${TASK_INDEX}-${log-short-name}.log
    """
    ma_vj_name_env = "MA_VJ_NAME"
    batch_job_id_env = 'BATCH_JOB_ID'

    vc_task_index_env = "VC_TASK_INDEX"
    vk_task_index_env = "VK_TASK_INDEX"

    ma_task_name_env = "MA_TASK_NAME"

    ma_current_ip_env = "MA_CURRENT_IP"

    log_file_ext = ".log"

    part_arr = []

    if ma_vj_name_env in os.environ:
        ma_vj_name = os.environ[ma_vj_name_env]
        if ma_vj_name is not None:
            part_arr.append(ma_vj_name.replace('ma-job', 'modelarts-job', 1))
    elif batch_job_id_env in os.environ:
        part_arr.append(os.environ[batch_job_id_env])

    if ma_task_name_env in os.environ:
        part_arr.append(os.environ[ma_task_name_env])

    if vc_task_index_env in os.environ:
        part_arr.append(os.environ[vc_task_index_env])
    elif vk_task_index_env in os.environ:
        part_arr.append(os.environ[vk_task_index_env])

    if ma_current_ip_env in os.environ:
        part_arr.append(os.environ[ma_current_ip_env])

    part_arr.append(log_short_name)

    return "-".join(part_arr) + log_file_ext


def get_tmp_log_dir():
    cache_path_env = "CACHE_HOME"
    mount_path_env = "MA_MOUNT_PATH"
    log_dir_name = "tmp-ascend-log"

    if cache_path_env in os.environ:
        return os.path.join(os.environ[cache_path_env], log_dir_name)

    if mount_path_env in os.environ:
        return os.path.join(os.environ[mount_path_env], log_dir_name)

    return os.path.join("/tmp", log_dir_name)


def tail_append_to_file(f, n, append_file):
    """
    :param f: original file
    :param n: tail n lines
    :param append_file: write n lines to a new file
    :return:
    """
    with open(append_file, 'a') as new_log_file:
        p = subprocess.Popen(['tail', '-n', str(n), f], stdout=new_log_file)
        p.wait()
    file_line_cnt = check_output(['wc', '-l', append_file])
    return int(file_line_cnt.split(b" ")[0])


if __name__ == '__main__':
    """
    python upload_tail_log.py -l 1024 -o s3://training-log/
    """
    args = init_parser()
    if args.lines is None or args.output is None:
        print("=== python upload_tail_log.py -l 1024 -o s3://training-log/ ===")
        exit(0)

    print_args(args)

    limited_log_lines = args.lines
    s3_upload_path = args.output

    tmp_log_dir = get_tmp_log_dir()
    os.makedirs(tmp_log_dir)

    collect_latest_n_log(tmp_log_dir, limited_log_lines)

    file_list = mox.file.list_directory(tmp_log_dir, recursive=True)
    file_num = len(file_list)
    print('totally, %d ascend log files to be uploaded' % file_num)

    mox.file.copy_parallel(tmp_log_dir, s3_upload_path, file_list=file_list)
