import os
import sys

import pyarrow.fs as fs

HDFS_URI = "viewfs://ftxz-hadoop-view/"
HDFS_USER = "zli"
REMOTE_ROOT = "/user/zli/lyt/data/StockIndexFutures/MSelect/FactorData"


def main():
    os.environ["HADOOP_USER_NAME"] = HDFS_USER
    hdfs = fs.HadoopFileSystem.from_uri(HDFS_URI)

    print("🔍 Scanning remote directory for date subdirectories...")
    infos = hdfs.get_file_info(fs.FileSelector(REMOTE_ROOT))
    date_dirs = sorted([info.path for info in infos if info.type == fs.FileType.Directory])

    output_file = "cal_weight_task_list.txt"
    with open(output_file, 'w') as fp:
        for dir_path in date_dirs:
            fp.write(dir_path + '\n')

    print(f"📝 Generated {len(date_dirs)} date directory tasks -> {output_file}")


if __name__ == "__main__":
    main()
