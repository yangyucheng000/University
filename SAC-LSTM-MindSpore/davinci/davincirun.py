import sys
import os

from common import SigHandler
from common import ModelArtsLog
from common import ModelArts
from common import RankTableEnv

from rank_table import RankTable, RankTableV0, RankTableV1
from sdr import RouteHelper

from manager import OpManager
from manager import SlogdManager
from manager import FMKManager
from manager import Manager
from manager import BatchLogManager
from manager import AscendVersionManager

if __name__ == '__main__':
    log = ModelArtsLog.setup_modelarts_logger()
    SigHandler.register_sig_child_handler()

    if len(sys.argv) <= 1:
        log.error('there are not enough args')
        sys.exit(1)

    batch_log_manager = BatchLogManager()
    batch_log_manager.run()

    op_manager = OpManager()
    return_code = op_manager.run()
    if return_code != 0:
        sys.exit(return_code)
    op_manager.destroy()

    AscendVersionManager.print_ascend_driver_version()
    if not AscendVersionManager.is_atlas_c75_tr5():
        log.info('you are advised to use ASCEND_DEVICE_ID env instead of DEVICE_ID,'
                 ' as the DEVICE_ID env will be discarded in later versions')
        log.info('particularly, ${ASCEND_DEVICE_ID} == ${DEVICE_ID}, it\'s the logical device id')

    train_command = sys.argv[1:]
    log.info('Davinci training command')
    log.info(train_command)

    if os.environ.get(RankTableEnv.RANK_TABLE_FILE_V1) is not None:
        # notebook generated rank_table with new v1 format
        rank_table_path = os.environ.get(RankTableEnv.RANK_TABLE_FILE_V1)
        RankTable.wait_for_available(rank_table_path)
        rank_table = RankTableV1(rank_table_path)
    else:
        # cce generated rank_table with old format
        rank_table_path_origin = RankTableEnv.get_rank_table_file_path()
        RankTable.wait_for_available(rank_table_path_origin)
        rank_table = RankTableV0(rank_table_path_origin)

    RankTableEnv.set_rank_table_env(rank_table.get_rank_table_path())

    slogd_manager = SlogdManager()
    slogd_manager.run()

    instance = rank_table.get_current_instance()
    server = rank_table.get_server(instance.server_id)
    # compare with instance, current_instance append rank_id in device
    current_instance = RankTable.convert_server_to_instance(server)

    new_current_instance = RouteHelper().do_route_plan(rank_table.get_rank_table_path(), instance)
    if new_current_instance is not None:
        current_instance = new_current_instance

    # TODO: delete it when new Ascend910 ModelArts Algorithms release or
    # AlgoRancher support Ascend910 v1 training mode
    # only keep special channel (data_url, train_url) in v1 format (for ModelArts Algorithm)
    ModelArts.only_keep_v1_special_channel_env()

    fmk_manager = FMKManager(current_instance)
    fmk_manager.run(rank_table.get_device_num(), train_command)
    return_code = fmk_manager.monitor()

    fmk_manager.destroy()
    Manager.destroy()
    batch_log_manager.destroy()

    sys.exit(return_code)
