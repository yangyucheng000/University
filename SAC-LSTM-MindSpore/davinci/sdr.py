import json
import os
import sys
import base64
import gzip
import time

from common import ModelArtsLog
from common import RankTableEnv
from rank_table import Device, Instance

log = ModelArtsLog.get_modelarts_logger()

TOPO_FILE_PATH = os.getenv('TOPO_FILE_PATH', '/user/config/ranktable_tor.json')
SDR_LIBRARY_PATH = '/usr/local/route'


class RouteHelper:
    def do_route_plan(self, rank_file, instance):
        # Default disable route plan acceleration.
        # It's allowed to set env ROUTE_PLAN = true to start the acceleration.
        route_plan_switch = os.getenv('ROUTE_PLAN', "false")
        if route_plan_switch.lower() == "false":
            log.info("Route plan ends for env ROUTE_PLAN = {}. "
                     "Route plan acceleration service may not be available "
                     "in this region".format(route_plan_switch))
            return None

        # Check python version. It requires to be python3.
        if sys.version_info[0] < 3:
            log.info("Route Plan ends with only support for Python 3 now.")
            return None

        # Check if route plan so files path is available.
        if os.path.exists(SDR_LIBRARY_PATH) is None:
            log.error("Cannot find software defined route module path.")
            return None
        sys.path.append(SDR_LIBRARY_PATH)

        # Check if route plan so code files exist.
        try:
            from route_plan import RoutePlan
        except ModuleNotFoundError:
            log.error("Route plan so files not found. "
                      "Please check files in {}".format(SDR_LIBRARY_PATH))
            return None

        log.info('Route plan begins. Current server {}'.format(
            instance.server_id))

        # Decompress topo file by base64 and gzip
        try:
            output_topo_file_path = self.decompress_topo_file(TOPO_FILE_PATH)
        except Exception as exception:
            log.error("Decompress route plan topo file error: {} : {}".format(
                type(exception).__name__, exception))

            return None

        save_rank_file = os.path.join(
            os.path.dirname(rank_file), "jobstart_routeplan.json")
        # Start route plan acceleration.
        try:
            ret, customdev, custom_id = RoutePlan.init(
                topo_file=output_topo_file_path,
                rank_file=rank_file,
                save_rank_file=save_rank_file,
                cursrvip=instance.server_id)

        # Catch every exception to avoid unexpected exit.
        except Exception as exception:
            self.log_exception(exception)
            return None

        if ret is False:
            log.info("Route plan ends. Route plan acceleration {}".format(ret))
            return None

        # Set and use the new route plan ranktable.
        RankTableEnv.set_rank_table_env(save_rank_file)
        self.log_ret(ret)
        devices = []
        for dev2 in customdev:
            dev = Device(dev2[1], dev2[2], dev2[0])
            devices.append(dev)
        inst = Instance("", custom_id, [])
        inst.set_devices(devices)
        current_instance = inst
        log.info('Finish route plan instance info.')
        return current_instance

    @staticmethod
    def wait_for_topo_available(input_topo_file_path, period=1):
        log.info('Wait for Rank table file ready')

        wait_time = 0
        maximum_wait_time = os.getenv('ROUTE_TOPO_WAIT_TIME', 60 * 5)
        decompressed_topo_string = ""

        while True:
            wait_time += period

            if wait_time > maximum_wait_time:
                log.info("Route plan wait time reaches the maximum {}s "
                         "for generating topo file {}".format(
                          maximum_wait_time, input_topo_file_path))
                return decompressed_topo_string

            if not os.path.exists(input_topo_file_path):
                time.sleep(period)
                continue
            try:
                with open(input_topo_file_path, 'r') as input_topo_file:
                    compressed_topo_string = input_topo_file.read()
                    decompressed_topo_string = gzip.decompress(
                        base64.b64decode(compressed_topo_string)).decode(
                        'utf-8')
                    topojson = json.loads(decompressed_topo_string)
                    if topojson["status"] == "completed":
                        return decompressed_topo_string
            except Exception as exception:
                log.debug("Route plan topo file {} is not available with compressed content {}"
                          .format(input_topo_file_path, compressed_topo_string),
                          exception)

            time.sleep(period)

        return decompressed_topo_string

    @staticmethod
    def decompress_topo_file(input_topo_file_path):

        decompressed_topo_string = RouteHelper.wait_for_topo_available(
            input_topo_file_path)

        log.info("Route plan decompress topo file as {}".format(
            decompressed_topo_string))

        output_topo_file_path = os.path.join(
            os.path.dirname(os.getcwd()),
            "ranktable_tor_decompress.json")

        with os.fdopen(os.open(os.path.realpath(output_topo_file_path),
                               os.O_WRONLY | os.O_CREAT, 0o640),
                       'w') as output_topo_file:
            output_topo_file.write(decompressed_topo_string)
            output_topo_file.close()

        return output_topo_file_path

    @staticmethod
    def log_exception(exception):
        # Default disable exception traceback.
        # It's allowed to set DEBUG_ROUTE_PLAN = TRUE to
        # enable exception traceback.
        if os.getenv('DEBUG_ROUTE_PLAN', "false").lower() == "true":
            debug_route_plan = True
        else:
            debug_route_plan = False

        log.error("Route plan failed for Exception [{}]: {}. "
                  "You are advised to set DEBUG_ROUTE_PLAN = true "
                  "in ma-pre-start.sh to see exception traceback.".format(
            type(exception).__name__, exception),
            exc_info=debug_route_plan)

    @staticmethod
    def log_ret(ret):
        log.info("Route plan ends. Route plan acceleration {}. "
                 "If you don't want route plan acceleration, "
                 "you can set ROUTE_PLAN = false in ma-pre-start.sh "
                 "to close it.".format(ret))