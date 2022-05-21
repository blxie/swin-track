import os
import numpy as np
import shutil
from data.operator.bbox.spatial.np.xyxy2xywh import bbox_xyxy2xywh
from miscellanies.torch.distributed import is_main_process
r"""
由于只是调试测试，这里只选取 GOT-10k 来进行测试，具体要修改下 datasets/easy_builder/builder.py 中的代码
我的目的是：测试的时候可视化测试结果

目前的问题是：当前类中只给出 datasets 的名称，没有具体的测试的图像 Tensor 数据

目前已知的条件：预测的 bbox 坐标，当前预测的视频序列

目前的解决思路：
  - 通过调试找到 predicted_bboxes 是在哪一个模块中被实现的（被定义的）
  - 由于是测试，因此 predicted_bboxes 的产生必然依赖于 tracker, datasets---GOT-10k
  - 目前推测在 train/test 相关模块中，这是起点，通过断点调试，以及 call stack 得到调用的整体流程
"""


class Got10kFormatPacker:

    def __init__(self, tracker_name, datasets, saving_path):
        self.tracker_name = tracker_name
        self.datasets = datasets
        self.saving_path = saving_path

    def save_sequence_result(self, dataset_unique_id, sequence_name,
                             predicted_bboxes, time_cost_array):
        sequence_path = os.path.join(self.saving_path, dataset_unique_id,
                                     self.tracker_name, sequence_name)
        os.makedirs(sequence_path, exist_ok=True)
        # 将预测的目标位置转化为点+偏移的表达形式
        predicted_bboxes = bbox_xyxy2xywh(predicted_bboxes)
        np.savetxt(
            os.path.join(sequence_path, f"{sequence_name}_001.txt"),
            predicted_bboxes,
            fmt="%.3f",
            delimiter=",",
        )
        np.savetxt(
            os.path.join(sequence_path, f"{sequence_name}_time.txt"),
            time_cost_array,
            fmt="%.8f",
        )

    def pack_dataset_result(self, dataset_unique_id):
        archive_base_path = os.path.join(
            self.saving_path, f"{self.tracker_name}-{dataset_unique_id}")
        shutil.make_archive(archive_base_path, "zip",
                            os.path.join(self.saving_path, dataset_unique_id))


class TrackingNetFormatPacker:

    def __init__(self, tracker_name, datasets, saving_path):
        self.tracker_name = tracker_name
        self.datasets = datasets
        self.saving_path = saving_path

    def save_sequence_result(self, dataset_unique_id, sequence_name,
                             predicted_bboxes, time_cost_array):
        sequence_path = os.path.join(self.saving_path, dataset_unique_id,
                                     self.tracker_name, sequence_name)
        os.makedirs(sequence_path, exist_ok=True)
        predicted_bboxes = bbox_xyxy2xywh(predicted_bboxes)
        np.savetxt(
            os.path.join(sequence_path, f"{sequence_name}.txt"),
            predicted_bboxes,
            fmt="%.2f",
            delimiter=",",
        )

    def pack_dataset_result(self, dataset_unique_id):
        archive_base_path = os.path.join(
            self.saving_path, f"{self.tracker_name}-{dataset_unique_id}")
        shutil.make_archive(archive_base_path, "zip",
                            os.path.join(self.saving_path, dataset_unique_id))


class TestOnlyDatasetTrackingResultSaver:

    def __init__(self, tracker_name, datasets, saving_path, packing_fn):
        self.tracker_name = tracker_name
        self.datasets = datasets
        self.saving_path = saving_path
        self.packing_fn = packing_fn
        self.run_results = {}
        self.cached_done_list = []

    def accept_evaluated_sequence(self, collected_sequences, io_thread):
        if collected_sequences is not None and len(collected_sequences) > 0:
            for (
                    dataset_unique_id,
                    sequence_name,
                    object_existence,
                    groundtruth_bboxes,
                    predicted_bboxes,
                    time_cost_array,
            ) in collected_sequences:
                if self.saving_path is not None and io_thread is not None:
                    io_thread.put((
                        self._save_sequence_predicted_bboxes,
                        dataset_unique_id,
                        sequence_name,
                        predicted_bboxes,
                        time_cost_array,
                    ))
                self.cached_done_list.append(
                    (dataset_unique_id, sequence_name))

    def prepare_gathering(self):
        return self.cached_done_list

    def accept_gathered(self, gathered_done_list, io_thread):
        for dataset_unique_id, sequence_name in gathered_done_list:
            if dataset_unique_id not in self.run_results:
                self.run_results[dataset_unique_id] = []
            self.run_results[dataset_unique_id].append(sequence_name)

            if is_main_process():
                if self.saving_path is not None and io_thread is not None:
                    if len(self.run_results[dataset_unique_id]) == len(
                            self.datasets[dataset_unique_id]):
                        io_thread.put((self.packing_fn.pack_dataset_result,
                                       dataset_unique_id))

        self.cached_done_list.clear()

    def _save_sequence_predicted_bboxes(self, dataset_unique_id, sequence_name,
                                        predicted_bboxes, time_cost_array):
        saving_path = os.path.join(self.saving_path, dataset_unique_id,
                                   sequence_name)
        os.makedirs(saving_path, exist_ok=True)
        assert len(predicted_bboxes) == len(time_cost_array)
        np.save(os.path.join(saving_path, "bounding_box.npy"),
                predicted_bboxes)
        np.save(os.path.join(saving_path, "time.npy"), time_cost_array)
        # TRACED
        self.packing_fn.save_sequence_result(dataset_unique_id, sequence_name,
                                             predicted_bboxes, time_cost_array)

    def get_summary(self):
        return None
