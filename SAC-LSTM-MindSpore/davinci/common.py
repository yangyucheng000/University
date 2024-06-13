import logging
import os
import signal
import errno
import json
import socket

logo = 'ModelArts'


# Batch env
class BatchEnv:
    BATCH_JOB_ID = 'BATCH_JOB_ID'
    POD_NAME = 'BATCH_TASK_CURRENT_INSTANCE'
    BATCH_TASK_LOG_PATH = 'BATCH_TASK_LOG_PATH'

    @staticmethod
    def get_pod_name():
        if BatchEnv.POD_NAME in os.environ and os.environ[BatchEnv.POD_NAME]:
            return os.environ[BatchEnv.POD_NAME]

        return None

    @staticmethod
    def get_batch_stdout_log():
        if BatchEnv.BATCH_TASK_LOG_PATH in os.environ and os.environ[BatchEnv.BATCH_TASK_LOG_PATH]:
            return os.path.join(os.environ[BatchEnv.BATCH_TASK_LOG_PATH], 'stdout.log')

        return None


# Rank Table Constants
class RankTableEnv:
    RANK_TABLE_FILE = 'RANK_TABLE_FILE'
    RANK_TABLE_FILE_V1 = 'RANK_TABLE_FILE_V_1_0'

    # jobstart_hccl.json is provided by the ring-controller of Cloud-Container-Engine(CCE)
    HCCL_JSON_FILE_NAME = 'jobstart_hccl.json'

    RANK_TABLE_FILE_DEFAULT_VALUE = '/user/config/%s' % HCCL_JSON_FILE_NAME

    @staticmethod
    def get_rank_table_v1_file_dir():
        parent_dir = '/home/work/'
        if ModelArts.MA_MOUNT_PATH_ENV in os.environ:
            parent_dir = os.environ[ModelArts.MA_MOUNT_PATH_ENV]
        if ModelArts.MA_HOME_ENV in os.environ:
            parent_dir = os.environ[ModelArts.MA_HOME_ENV]

        return os.path.join(parent_dir, 'rank_table')

    @staticmethod
    def get_rank_table_file_path():
        rank_table_file_path = os.environ.get(RankTableEnv.RANK_TABLE_FILE)
        if rank_table_file_path is None:
            return RankTableEnv.RANK_TABLE_FILE_DEFAULT_VALUE

        return os.path.join(os.path.normpath(rank_table_file_path), RankTableEnv.HCCL_JSON_FILE_NAME)

    @staticmethod
    def set_rank_table_env(path):
        os.environ[RankTableEnv.RANK_TABLE_FILE] = path


class OpEnv:
    MA_ALGORITHM_OPERATOR = 'MA_ALGORITHM_OPERATOR'
    OBS_TYPE = 'obs'
    OBS_URL = 'obs_url'
    IDE_MODE = 'ide_mode'

    @staticmethod
    def should_handle_operator():
        if OpEnv.MA_ALGORITHM_OPERATOR in os.environ and os.environ[OpEnv.MA_ALGORITHM_OPERATOR] != '':
            op_str = os.environ[OpEnv.MA_ALGORITHM_OPERATOR]
            op_json_info = json.loads(op_str)
            if OpEnv.OBS_TYPE in op_json_info and OpEnv.OBS_URL in op_json_info[OpEnv.OBS_TYPE]:
                return True
        return False

    @staticmethod
    def get_op_obs_uri():
        if OpEnv.should_handle_operator():
            op_str = os.environ[OpEnv.MA_ALGORITHM_OPERATOR]
            op_json_info = json.loads(op_str)
            return op_json_info[OpEnv.OBS_TYPE][OpEnv.OBS_URL]
        return None

    @staticmethod
    def ide_mode():
        if OpEnv.should_handle_operator():
            op_str = os.environ[OpEnv.MA_ALGORITHM_OPERATOR]
            op_json_info = json.loads(op_str)
            if OpEnv.IDE_MODE in op_json_info and op_json_info[OpEnv.IDE_MODE]:
                return True
        return False


class ModelArts:
    MA_MODELARTS_DOWNLOADER = '/home/work/modelarts-downloader.py'

    tmp_log_dir = '/tmp/log/'

    modelarts_pipe_cmd = 'tee'

    MA_HOME_ENV = 'MA_HOME'
    MA_MOUNT_PATH_ENV = 'MA_MOUNT_PATH'
    MA_CURRENT_INSTANCE_NAME_ENV = 'MA_CURRENT_INSTANCE_NAME'
    MA_VJ_NAME = 'MA_VJ_NAME'
    MA_JOB_KIND = 'MA_JOB_KIND'

    MA_INPUTS = 'MA_INPUTS'
    MA_OUTPUTS = 'MA_OUTPUTS'

    MA_UPLOAD_LOG_OBS_ENV = 'DLS_UPLOAD_LOG_OBS_DIR'
    MA_USE_UPLOADER_ENV = 'DLS_USE_UPLOADER'

    MA_CURRENT_HOST_IP = 'MA_CURRENT_HOST_IP'

    @staticmethod
    def get_current_instance_name():
        if BatchEnv.POD_NAME in os.environ:
            # v1
            return os.environ[BatchEnv.POD_NAME]

        if ModelArts.MA_CURRENT_INSTANCE_NAME_ENV in os.environ:
            # v2
            return os.environ[ModelArts.MA_CURRENT_INSTANCE_NAME_ENV]

        # edge
        return None

    @staticmethod
    def get_current_host_ip():
        return os.environ.get(ModelArts.MA_CURRENT_HOST_IP)

    @staticmethod
    def get_job_id():
        if BatchEnv.BATCH_JOB_ID in os.environ:
            return os.environ[BatchEnv.BATCH_JOB_ID]

        if ModelArts.MA_VJ_NAME in os.environ:
            ma_vj_name = os.environ[ModelArts.MA_VJ_NAME]
            return ma_vj_name.replace('ma-job', 'modelarts-job', 1)

        return socket.gethostname()

    @staticmethod
    def enable_log_upload():
        if ModelArts.MA_UPLOAD_LOG_OBS_ENV in os.environ and os.environ[ModelArts.MA_UPLOAD_LOG_OBS_ENV] and \
                ModelArts.MA_USE_UPLOADER_ENV in os.environ:
            return True

        return False

    @staticmethod
    def get_log_upload_url():
        if ModelArts.enable_log_upload():
            return os.environ[ModelArts.MA_UPLOAD_LOG_OBS_ENV]

        return None

    @staticmethod
    def is_edge_job():
        if ModelArts.MA_JOB_KIND in os.environ and os.environ[ModelArts.MA_JOB_KIND] == 'edge_job':
            return True

        return False

    @staticmethod
    def generate_v1_channel_from_v2_channel(channel_type, v2_channel, special_parameter):
        if channel_type in v2_channel:
            for channel in v2_channel[channel_type]:
                if channel.get("name") == special_parameter:
                    data_source = ''
                    obs_url = ''
                    if "datasets" in channel["remote"]:
                        data_source = "datasets"
                        obs_url = channel["remote"]["datasets"]["obs_url"]
                    elif "obs" in channel["remote"]:
                        data_source = "obs"
                        obs_url = channel["remote"]["obs"]["obs_url"]

                    s3_protocol = 's3:/'
                    if obs_url.startswith(s3_protocol):
                        obs_url = obs_url[len(s3_protocol):]

                    parameter_value = channel["local_dir"]
                    channel_v1_template = '{"%s": [{"parameter": {"label": "%s", "value": "%s"}' \
                                          ', "data_source": {"%s": {"obs_url": "%s"}}}]}' % \
                                          (channel_type, special_parameter, parameter_value, data_source, obs_url)
                    return channel_v1_template

        return None

    @staticmethod
    def only_keep_v1_special_channel_env():
        if ModelArts.MA_INPUTS in os.environ and os.environ[ModelArts.MA_INPUTS] != "":
            input_json = json.loads(os.environ[ModelArts.MA_INPUTS])
            v1_special_input_json = ModelArts.generate_v1_channel_from_v2_channel("inputs", input_json, "data_url")
            if v1_special_input_json is not None:
                os.environ[ModelArts.MA_INPUTS] = v1_special_input_json
            else:
                del os.environ[ModelArts.MA_INPUTS]

        if ModelArts.MA_OUTPUTS in os.environ and os.environ[ModelArts.MA_OUTPUTS] != "":
            output_json = json.loads(os.environ[ModelArts.MA_OUTPUTS])
            v1_special_output_json = ModelArts.generate_v1_channel_from_v2_channel("outputs", output_json, "train_url")
            if v1_special_output_json is not None:
                os.environ[ModelArts.MA_OUTPUTS] = v1_special_output_json
            else:
                del os.environ[ModelArts.MA_OUTPUTS]


class HwHiAiUser:
    PRE_STOP_SCRIPTS = '/usr/local/Ascend/driver/tools/docker_stop_post_sys.sh'

    MIND_STUDIO_OP_DIR = 'operator_info'

    FMK_WORKSPACE_ENV = 'FMK_WORKSPACE'

    FMK_WORKSPACE = 'workspace'
    FMK_WORKSPACE_DEFAULT_VALUE = os.path.join('/home/work', FMK_WORKSPACE)

    HCCL_CONNECT_TIMEOUT = 'HCCL_CONNECT_TIMEOUT'

    ASCEND_GLOBAL_LOG_LEVEL = 'ASCEND_GLOBAL_LOG_LEVEL'
    ASCEND_GLOBAL_EVENT_ENABLE = 'ASCEND_GLOBAL_EVENT_ENABLE'

    ASCEND_SLOG_PRINT_TO_STDOUT = 'ASCEND_SLOG_PRINT_TO_STDOUT'

    MINDSPORE_FRAMEWORK_NAME = 'mindspore'
    MINDSPORE_FRAMEWORK_FAULTS_DIAG_VERSION = '1.4'

    @staticmethod
    def get_fmk_workspace_dir():
        if ModelArts.MA_MOUNT_PATH_ENV in os.environ:
            return os.path.join(os.environ.get(ModelArts.MA_MOUNT_PATH_ENV), HwHiAiUser.FMK_WORKSPACE)

        return os.environ.get(HwHiAiUser.FMK_WORKSPACE_ENV, HwHiAiUser.FMK_WORKSPACE_DEFAULT_VALUE)


class ModelArtsLog:

    @staticmethod
    def setup_modelarts_logger():
        name = logo
        formatter = logging.Formatter(fmt='[ModelArts Service Log]%(asctime)s - %(levelname)s - %(message)s')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    @staticmethod
    def get_modelarts_logger():
        return logging.getLogger(logo)


_registered_child_ids = []


class SigHandler:

    @staticmethod
    def register_sig_child_handler():
        signal.signal(signal.SIGCHLD, SigHandler.wait_child)

    @staticmethod
    def register_wait_child(child_pid):
        if child_pid not in _registered_child_ids:
            _registered_child_ids.append(child_pid)

    @staticmethod
    def wait_child(signum, frame):
        try:
            for child_pid in _registered_child_ids:
                _, status = os.waitpid(child_pid, os.WNOHANG)
        except OSError as e:
            if e.errno == errno.ECHILD:
                pass
            else:
                raise
