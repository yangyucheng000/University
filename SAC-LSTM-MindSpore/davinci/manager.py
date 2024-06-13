import time
import subprocess
import os
import os.path
import stat
import signal
import threading

from common import ModelArtsLog
from common import SigHandler
from common import ModelArts
from common import HwHiAiUser
from common import OpEnv
from common import BatchEnv
from fmk import FMK

import moxing as mox

log = ModelArtsLog.get_modelarts_logger()


class Manager:
    def __init__(self):
        pass

    @staticmethod
    def destroy():
        if os.path.isfile(HwHiAiUser.PRE_STOP_SCRIPTS):
            subprocess.call([HwHiAiUser.PRE_STOP_SCRIPTS])


class FMKManager:
    # max destroy time: ~20 (15 + 5)
    # ~ 15 (1 + 2 + 4 + 8)
    MAX_TEST_PROC_CNT = 4
    KILL_WAIT_TIME = 5

    def __init__(self, instance):
        self.instance = instance
        self.fmk = []
        self.fmk_processes = []
        self.get_sigterm = False

    # break the monitor and destory processes when get terminate signal
    def term_handle(func):
        def receive_term(signum, stack):
            log.info('Received terminate signal %d, try to destroyed all processes' % signum)
            stack.f_locals['self'].get_sigterm = True

        def handle_func(self, *args, **kwargs):
            origin_handle = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, receive_term)
            res = func(self, *args, **kwargs)
            signal.signal(signal.SIGTERM, origin_handle)
            return res

        return handle_func

    def run(self, rank_size, command):
        c75_tr5_flag = AscendVersionManager.is_atlas_c75_tr5()
        for index, device in enumerate(self.instance.devices):
            fmk_instance = FMK(c75_tr5_flag, index, device)
            self.fmk.append(fmk_instance)

            self.fmk_processes.append(fmk_instance.run(rank_size, command))

    @term_handle
    def monitor(self, period=1):
        # busy waiting for all fmk processes exit by zero
        # or there is one process exit by non-zero

        fmk_cnt = len(self.fmk_processes)
        zero_ret_cnt = 0
        while zero_ret_cnt != fmk_cnt:
            zero_ret_cnt = 0
            for index in range(fmk_cnt):
                fmk = self.fmk[index]
                fmk_process = self.fmk_processes[index]
                if fmk_process.poll() is not None:
                    if fmk_process.returncode != 0:
                        log.error('proc-rank-%s-device-%s (pid: %d) has exited with non-zero code: %d'
                                  % (fmk.rank_id, fmk.device_id, fmk_process.pid, fmk_process.returncode))
                        return fmk_process.returncode

                    zero_ret_cnt += 1
            if self.get_sigterm:
                break
            time.sleep(period)

        return 0

    def destroy(self, base_period=1):
        log.info('Begin destroy training processes')
        self.send_sigterm_to_fmk_process()
        self.wait_fmk_process_end(base_period)
        log.info('End destroy training processes')

    def send_sigterm_to_fmk_process(self):
        # send SIGTERM to fmk processes (and process group)
        for r_index in range(len(self.fmk_processes) - 1, -1, -1):
            fmk = self.fmk[r_index]
            fmk_process = self.fmk_processes[r_index]
            if fmk_process.poll() is not None:
                log.info('proc-rank-%s-device-%s (pid: %d) has exited', fmk.rank_id, fmk.device_id, fmk_process.pid)
                del self.fmk_processes[r_index]
                del self.fmk[r_index]

            try:
                os.killpg(fmk_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def wait_fmk_process_end(self, base_period):
        test_cnt = 0
        period = base_period
        while len(self.fmk_processes) > 0 and test_cnt < self.MAX_TEST_PROC_CNT:
            for r_index in range(len(self.fmk_processes) - 1, -1, -1):
                fmk = self.fmk[r_index]
                fmk_process = self.fmk_processes[r_index]
                if fmk_process.poll() is not None:
                    log.info('proc-rank-%s-device-%s (pid: %d) has exited',
                             fmk.rank_id, fmk.device_id, fmk_process.pid)
                    del self.fmk_processes[r_index]
                    del self.fmk[r_index]

            time.sleep(period)
            period *= 2
            test_cnt += 1

        if len(self.fmk_processes) > 0:
            for r_index in range(len(self.fmk_processes) - 1, -1, -1):
                fmk = self.fmk[r_index]
                fmk_process = self.fmk_processes[r_index]
                if fmk_process.poll() is None:
                    log.warn('proc-rank-%s-device-%s (pid: %d) has not exited within the max waiting time, '
                             'send kill signal',
                             fmk.rank_id, fmk.device_id, fmk_process.pid)
                    os.killpg(fmk_process.pid, signal.SIGKILL)


class SlogdManager:
    def __init__(self, executable_path='/usr/local/Ascend/driver/tools/docker/slogd'):
        self.bin_path = executable_path if os.path.isfile(executable_path) else None
        self.slogd_process = None

    def run(self):
        if self.bin_path is None:
            # there is no slogd anymore after C76
            return

        log.info('Slogd startup')
        self.slogd_process = subprocess.Popen([self.bin_path])

        # slogd attach itself to PPID 0
        # and the child process will be terminated
        # this can avoid: [slogd] <defunct>
        SigHandler.register_wait_child(self.slogd_process.pid)


class OpManager:
    def __init__(self, op_workspace='/tmp'):
        self.op_workspace = op_workspace

    # download archive from obs to dest_path (call modelarts-downloader.py)
    @staticmethod
    def download_from_obs(raw_obs_url, dest_path):
        download_cmd = 'python %s -s %s -d %s'
        if raw_obs_url.endswith(os.path.sep):
            # recursive download the content of dir (object) and skip creating the dir
            download_cmd += ' --skip-creating-dir -r'

        command = download_cmd % (ModelArts.MA_MODELARTS_DOWNLOADER, raw_obs_url, dest_path)
        return subprocess.Popen(command, shell=True).wait()

    # exec xxx.run
    @staticmethod
    def install_op(op_path):
        op_run_file = None
        for file_name in os.listdir(op_path):
            if file_name.endswith('.run'):
                op_run_file = file_name
                break

        if not op_run_file:
            log.error("operator archive not found")
            return 255

        op_run = '%s/%s' % (op_path, op_run_file)
        # +x
        op_run_st = os.stat(op_run)
        os.chmod(op_run, op_run_st.st_mode | stat.S_IEXEC)

        # exec xxx.run
        return subprocess.Popen(op_run, shell=True).wait()

    def run(self):
        if not OpEnv.should_handle_operator():
            return 0

        log.info('download the operator archive from obs')
        tmp_op_path = '%s/%s' % (self.op_workspace, HwHiAiUser.MIND_STUDIO_OP_DIR)
        os.makedirs(tmp_op_path)

        return_code = OpManager.download_from_obs(OpEnv.get_op_obs_uri(), tmp_op_path)
        if return_code != 0:
            log.error('download the operator archive from obs failed, return code: [%d]' % return_code)
            return return_code

        return_code = OpManager.install_op(tmp_op_path)
        if return_code != 0:
            log.error('install the operator failed, return code: [%d]' % return_code)
            return return_code

        return 0

    # clear tmp op archive
    def destroy(self):
        return subprocess.Popen('rm -rf %s/%s' % (self.op_workspace, HwHiAiUser.MIND_STUDIO_OP_DIR), shell=True).wait()


class BatchLogManager:

    def __init__(self, upload_interval=30, upload_time_warning_threshold=8):
        self.local_stdout_log_path = None
        self.obs_log_url = None

        self.upload_interval = upload_interval

        self.background_uploader_thread = None
        self.ticker = threading.Event()

        self.upload_time_warning_threshold = upload_time_warning_threshold

    def run(self):
        if not ModelArts.enable_log_upload() or not BatchEnv.get_batch_stdout_log() or not BatchEnv.get_pod_name():
            return

        if self.background_uploader_thread is not None:
            return

        self.local_stdout_log_path = BatchEnv.get_batch_stdout_log()
        if not os.path.isfile(self.local_stdout_log_path):
            log.warn('stdout log %s is not found' % self.local_stdout_log_path)
            return

        self.obs_log_url = os.path.join(ModelArts.get_log_upload_url(), BatchLogManager.get_obs_log_file_name())
        log.info('background upload stdout log to %s' % self.obs_log_url)

        self.background_uploader_thread = threading.Thread(target=BatchLogManager.background_upload_log_to_obs,
                                                           args=(self.ticker,
                                                                 self.upload_interval,
                                                                 self.upload_time_warning_threshold,
                                                                 self.local_stdout_log_path,
                                                                 self.obs_log_url))
        self.background_uploader_thread.start()

    @staticmethod
    def get_obs_log_file_name():
        return BatchEnv.get_pod_name() + '.log'

    @staticmethod
    def background_upload_log_to_obs(ticker, upload_interval, upload_time_warning_threshold,
                                     local_stdout_log_path, obs_log_url):
        upload_time_is_too_long_warning = False

        while not ticker.wait(upload_interval):
            if not upload_time_is_too_long_warning:
                start_time = time.time()
                BatchLogManager.upload_log_to_obs(local_stdout_log_path, obs_log_url)
                upload_time = time.time() - start_time
                if upload_time > upload_time_warning_threshold:
                    log.warn('upload stdout log time is larger than %s seconds, log file size %s',
                             upload_time_warning_threshold, os.path.getsize(local_stdout_log_path))
                    upload_time_is_too_long_warning = True
            else:
                BatchLogManager.upload_log_to_obs(local_stdout_log_path, obs_log_url)

        # at exit
        BatchLogManager.upload_log_to_obs(local_stdout_log_path, obs_log_url)
        log.info('final upload stdout log done')

    @staticmethod
    def upload_log_to_obs(local_stdout_log_path, obs_log_url):
        mox.file.copy(local_stdout_log_path, obs_log_url)

    def destroy(self):
        if self.background_uploader_thread is None:
            return

        self.ticker.set()

        # stdout.log size is limited under 1Gb by modelarts-pipe
        # generally, it can be uploaded within 60s
        # 180 timeout = 60 (upload_log_to_obs time) + 60 (upload_log_to_obs time) + 60 (reserve time)
        self.background_uploader_thread.join(180)


class AscendVersionManager:
    driver_version_file_path = '/usr/local/Ascend/driver/version.info'

    c75_tr5_driver_version = 'Version=20.1.0'

    @staticmethod
    def test_driver_version_file_exists():
        return os.path.isfile(AscendVersionManager.driver_version_file_path)

    @staticmethod
    def print_ascend_driver_version():
        if not AscendVersionManager.test_driver_version_file_exists():
            log.warn('there is no %s file' % AscendVersionManager.driver_version_file_path)
            log.info('Ascend Driver: Unknown')
            return

        with open(AscendVersionManager.driver_version_file_path) as version_file:
            for line in version_file:
                line = line.strip()
                log.info('Ascend Driver: %s' % line)
                # we only take the first line into account
                return

        return

    @staticmethod
    def is_atlas_c75_tr5():
        if not AscendVersionManager.test_driver_version_file_exists():
            return False

        with open(AscendVersionManager.driver_version_file_path) as version_file:
            for line in version_file:
                line = line.strip()
                if line == AscendVersionManager.c75_tr5_driver_version:
                    return True
                return False

        return False
