import os
import subprocess
import pathlib
from contextlib import contextmanager

from common import ModelArtsLog
from common import ModelArts
from common import BatchEnv
from common import HwHiAiUser
from common import OpEnv

log = ModelArtsLog.get_modelarts_logger()


class FMK:

    def __init__(self, c75_tr5, index, device):
        self.c75_tr5 = c75_tr5

        self.job_id = ModelArts.get_job_id()
        self.rank_id = device.rank_id
        if not c75_tr5:
            # logic device id after c75-tr5
            # specially, mindspore needs logic device id in c75-tr5 and after
            self.device_id = str(index)
        else:
            # physical device id in c75-tr5 (and before)
            self.device_id = device.device_id

    def gen_env_for_fmk(self, rank_size):
        current_envs = os.environ.copy()
        current_envs['JOB_ID'] = self.job_id

        if not self.c75_tr5:
            # import a new ASCEND_DEVICE_ID env as the logical device id after c75-tr5
            current_envs['ASCEND_DEVICE_ID'] = self.device_id
        # the DEVICE_ID env will be deprecated, keep it in order to be compatible with moxing and mindspore
        # physical device id in c75-tr5 (non mindspore)
        # logical device id after c75-tr5
        current_envs['DEVICE_ID'] = self.device_id

        current_envs['RANK_ID'] = self.rank_id
        current_envs['RANK_SIZE'] = str(rank_size)

        FMK.set_env_if_not_exist(current_envs, HwHiAiUser.HCCL_CONNECT_TIMEOUT, str(1800))  # 30min

        if OpEnv.ide_mode():
            current_envs['SLOG_PRINT_TO_STDOUT'] = '1'

        self.gen_diag_mode_env(current_envs)

        return current_envs

    def gen_diag_mode_env(self, current_envs):
        log_dir = FMK.get_log_dir()
        process_log_path = os.path.join(log_dir, self.job_id, 'ascend', 'process_log', 'rank_' + self.rank_id)
        FMK.set_env_if_not_exist(current_envs, 'ASCEND_PROCESS_LOG_PATH', process_log_path)
        pathlib.Path(current_envs['ASCEND_PROCESS_LOG_PATH']).mkdir(parents=True, exist_ok=True)
        diag_mode = current_envs.get('MA_DIAG_MODE_ENV', '')
        run_mode = current_envs.get('MA_RUN_MODE_ENV', '')
        engine_version = current_envs.get("MA_ENGINE_VERSION", '')
        glog_dir = ms_rdr_path = ms_om_path = os.path.join(log_dir, self.job_id, 'mindspore', 'log')
        if diag_mode == 'faults':
            FMK.set_env_if_not_exist(current_envs, 'PRINT_MODEL', str(1))
            FMK.set_env_if_not_exist(current_envs, 'DUMP_GE_GRAPH', str(2))
            FMK.set_env_if_not_exist(current_envs, 'DUMP_GRAPH_LEVEL', str(2))
            FMK.set_env_if_not_exist(current_envs, 'ASCEND_GLOBAL_LOG_LEVEL', str(1))
            FMK.set_env_if_not_exist(current_envs, 'ASCEND_HOST_LOG_FILE_NUM', str(1000))

            npu_collect_path = os.path.join(log_dir, self.job_id, 'ascend', 'npu_collect', 'rank_' + self.rank_id)
            FMK.set_env_if_not_exist(current_envs, 'NPU_COLLECT_PATH', npu_collect_path)
            pathlib.Path(os.path.join(current_envs['NPU_COLLECT_PATH'], 'extra-info', 'graph')).mkdir(parents=True, exist_ok=True)

            framework_name_version = next(iter(engine_version.split('-')[0:1]), '')
            framework_version = next(iter(framework_name_version.split('_')[1:2]), '')
            if HwHiAiUser.MINDSPORE_FRAMEWORK_NAME in framework_name_version and HwHiAiUser.MINDSPORE_FRAMEWORK_FAULTS_DIAG_VERSION <= framework_version:
                FMK.set_env_if_not_exist(current_envs, 'GLOG_v', str(1))
                FMK.set_env_if_not_exist(current_envs, 'GLOG_log_dir', glog_dir)
                FMK.set_env_if_not_exist(current_envs, 'GLOG_logtostderr', str(0))
                FMK.set_env_if_not_exist(current_envs, 'MS_RDR_ENABLE', str(1))
                FMK.set_env_if_not_exist(current_envs, 'MS_RDR_PATH', ms_rdr_path)
                FMK.set_env_if_not_exist(current_envs, 'MS_OM_PATH', ms_om_path)

        elif diag_mode == 'accuracy' or diag_mode == 'profile':
            diag_data_path = os.path.join(log_dir, self.job_id, 'mindspore', 'diagnostic_data')
            FMK.set_env_if_not_exist(current_envs, 'MS_DIAGNOSTIC_DATA_PATH', diag_data_path)

        elif run_mode == 'performance':
            FMK.set_env_if_not_exist(current_envs, 'ASCEND_GLOBAL_LOG_LEVEL', str(3))
            FMK.set_env_if_not_exist(current_envs, 'ASCEND_GLOBAL_EVENT_LEVEL', str(0))
            FMK.set_env_if_not_exist(current_envs, 'GLOG_v', str(3))
            FMK.set_env_if_not_exist(current_envs, 'GLOG_log_dir', glog_dir)
            FMK.set_env_if_not_exist(current_envs, 'GLOG_logtostderr', str(0))
            FMK.set_env_if_not_exist(current_envs, 'MS_OM_PATH', ms_om_path)

        elif run_mode == 'normal':
            FMK.set_env_if_not_exist(current_envs, 'GLOG_v', str(1))
            FMK.set_env_if_not_exist(current_envs, 'GLOG_log_dir', glog_dir)
            FMK.set_env_if_not_exist(current_envs, 'GLOG_logtostderr', str(0))
            FMK.set_env_if_not_exist(current_envs, 'MS_OM_PATH', ms_om_path)

    @contextmanager
    def switch_directory(self, directory):
        owd = os.getcwd()
        try:
            os.chdir(directory)
            yield directory
        finally:
            os.chdir(owd)

    def get_working_dir(self):
        fmk_workspace_prefix = HwHiAiUser.get_fmk_workspace_dir()
        return os.path.join(os.path.normpath(fmk_workspace_prefix), 'device%s' % self.device_id)

    @staticmethod
    def get_log_dir():
        batch_log_path = os.getenv(BatchEnv.BATCH_TASK_LOG_PATH)
        if batch_log_path and os.path.exists(batch_log_path):
            return batch_log_path

        modelarts_mount_path = os.getenv(ModelArts.MA_MOUNT_PATH_ENV)
        if modelarts_mount_path:
            modelarts_log_path = os.path.join(modelarts_mount_path, 'log')
            if os.path.exists(modelarts_log_path):
                return modelarts_log_path

        return ModelArts.tmp_log_dir

    @staticmethod
    def set_env_if_not_exist(envs, env_name, env_value):
        if env_name in os.environ:
            log.info('env already exists. env_name: %s, env_value: %s ' % (env_name, env_value))
            return
        envs[env_name] = env_value

    def run(self, rank_size, command):
        envs = self.gen_env_for_fmk(rank_size)
        log.info('bootstrap proc-rank-%s-device-%s' % (self.rank_id, self.device_id))

        working_dir = self.get_working_dir()
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        log_dir = FMK.get_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if self.c75_tr5:
            with self.switch_directory(working_dir):
                return subprocess.Popen(command, env=envs, preexec_fn=os.setsid)

        # we `tee` a proc log of each training processes after c75-tr5

        # AOM collect (*.trace | *.log | *.out) log file
        # let log_file end with .txt, avoid AOM collect it
        log_file = '%s-proc-rank-%s-device-%s.txt' % (self.job_id, self.rank_id, self.device_id)
        log_file_path = os.path.join(log_dir, log_file)

        with self.switch_directory(working_dir):
            # os.setsid: change the process(forked) group id to itself
            training_proc = subprocess.Popen(command, env=envs, preexec_fn=os.setsid,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            log.info('proc-rank-%s-device-%s (pid: %d)', self.rank_id, self.device_id, training_proc.pid)

            # https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
            # modelarts_pipe_cmd should consume the stdout in time and avoid proc deadlock
            # and currently, we use `tee` instead of `modelarts-pipe`, as the modelarts-pipe requires singleton
            # TODO: limit the splitting log file size < 1GB
            subprocess.Popen([ModelArts.modelarts_pipe_cmd, log_file_path], stdin=training_proc.stdout)

            return training_proc
