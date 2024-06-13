# coding:utf-8
import argparse
import os
import re


def init_parser():
    # 解析parse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", help="start time")
    parser.add_argument("-e", "--end", help="end time")
    parser.add_argument("-o", "--output", help="output file")
    return parser.parse_args()


def parse_log(input_path, output_file, start_time, end_time):
    name_list = os.listdir(input_path)
    full_list = [os.path.join(input_path, i) for i in name_list]
    # list file sorted by date
    time_sorted_list = sorted(full_list, key=os.path.getmtime, reverse=True)
    latest_log = time_sorted_list[0]
    f_out = open(output_file, "w")
    # 暂时全部日志
    time_pattern = re.compile(r'\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}')
    # convert time from %Y-%m-%d-%H-%M-%S to %Y%m%d%H%M%S
    device_start_time = start_time.replace("-", "").replace(":", "")
    for device in time_sorted_list:
        try:
            with open(device, "r") as f:
                for line in f:
                    if start_time <= time_pattern.findall(line)[0] <= end_time:
                        f_out.write(line)
            # find the first device log file
            device_time = re.findall(r"\d{14}", device)[0]
            if device_time < device_start_time:
                break
        except:
            pass

    f_out.close()


if __name__ == '__main__':
    args = init_parser()

    DEVICE_PATH = "/device"
    output_path = str(args.output)
    if output_path.endswith("/"):
        output_path = output_path[:-1]
    """
           python splitLog.py -s start_time -e end_time -f output_file
    """
    if args.start is None:
        print("=== python splitLog.py -s start_time -e end_time -o output ... ===")
        exit(0)
    elif args.end is None:
        print("=== python splitLog.py -s start_time -e end_time -o output ... ===")
        exit(0)
    elif args.output is None:
        print("=== python splitLog.py -s start_time -e end_time -o output ... ===")
        exit(0)

    # 处理device 0~7的日志
    for index in range(0, 8):
        parse_log(DEVICE_PATH + "/device-" + str(index), output_path + "/device-" + str(index), args.start, args.end)
    # 处理device-os-0的日志
    parse_log(DEVICE_PATH + "/device-os-0", output_path + "/device-os-0", args.start, args.end)
    # 处理device-os-4的日志
    parse_log(DEVICE_PATH + "/device-os-4", output_path + "/device-os-4", args.start, args.end)
