import encodings
import numpy as np
import cv2
import os
from PIL import Image


def read_image(filename, color_fmt='RGB'):
    assert color_fmt in Image.MODES
    img = Image.open(filename)
    if not img.mode == color_fmt:
        img = img.convert(color_fmt)
    return np.asarray(img)


def save_image(filename, img, color_fmt='RGB'):
    assert color_fmt in ['RGB', 'BGR']
    if color_fmt == 'BGR' and img.ndim == 3:
        img = img[..., ::-1]
    img = Image.fromarray(img)
    return img.save(filename)


def show_image(
        img,
        bboxes=None,
        bbox_fmt='ltrb',
        colors=None,
        thickness=2,  # 线的粗细
        fig=1,
        delay=1,
        max_size=640,
        visualize=True,
        cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if bboxes is not None:
            bboxes = np.array(bboxes, dtype=np.float32) * scale

    if bboxes is not None:
        assert bbox_fmt in ['ltwh', 'ltrb']
        bboxes = np.array(bboxes, dtype=np.int32)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
        if bboxes.shape[1] == 4 and bbox_fmt == 'ltwh':
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:] - 1

        # clip bounding boxes
        h, w = img.shape[:2]
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)

        if colors is None:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                      (255, 0, 255), (255, 255, 0), (0, 0, 128), (0, 128, 0),
                      (128, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            if len(bbox) == 4:
                pt1 = (int(bbox[0]), int(bbox[1]))  # 左上
                pt2 = (int(bbox[2]), int(bbox[3]))  # 右下
                img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
            else:
                pts = bbox.reshape(-1, 2)
                img = cv2.polylines(img, [pts], True, color.tolist(),
                                    thickness)

    if visualize:
        if isinstance(fig, str):
            winname = fig
        else:
            winname = 'window_{}'.format(fig)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    if cvt_code in [cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2RGB]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def show_frames(sequence_dir, frame_format, sequence_len, predict_bboxes):
    """输出最后的显示结果
    sequence_dir: 序列的文件夹部分
    frame_format: 帧的格式 传入格式化字符串
      '{:08d}.jpg' 表示 GOT-10k 以及 LaSOT 的 frame 格式
      '{}.jpg' 表示 TrackingNet 的格式
    sequence_len: 序列的长度
      设置的原因是：LaSOT 里面有一个 groundtruth.txt，混合在里面
    predict_bboxes: 预测的边界框
    """
    for i in range(1, sequence_len):
        # (1) 获取 image
        img_file = os.path.join(sequence_dir, frame_format.format(i))
        # print(img_file)

        # (2) 获取 predict_bbox
        predict_bbox = predict_bboxes[i]
        # # (3) 展示添加预测框的图片
        # flags[0, 1] 分别表示灰度、彩色图像
        # img = cv2.imread(img_file, flags=1)
        img = read_image(filename=img_file, color_fmt='RGB')
        img = cv2.rectangle(
            img,
            (int(predict_bbox[0]), int(predict_bbox[1])),
            (int(predict_bbox[0] + predict_bbox[2]),
             int(predict_bbox[1] + predict_bbox[3])),
            (0, 0, 255),
            3,
        )

        if max(img.shape[:2]) > 640:
            scale = 640 / max(img.shape[:2])
            out_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, out_size)
        cv2.imshow("TEST", img)
        cv2.waitKey(10)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for n in range(160, 181):
        sequence_dir = '/data/GOT-10k/test/GOT-10k_Test_{:06d}/'.format(n)
        predict_bboxes = []
        with open(
                "/home/guest/XieBailian/proj/SwinTrack/visualize/got-10k-test/Tiny/GOT-10k_Test_{:06d}/GOT-10k_Test_{:06d}_001.txt"
                .format(n, n)) as fp:
            lines = fp.readlines()
            # print(lines)
            for line in lines:
                predict_bbox = np.fromstring(line, dtype=float, sep=",")
                predict_bboxes.append(predict_bbox)

        show_frames(sequence_dir, '{:08d}.jpg',
                    len(os.listdir(sequence_dir)) - 1, predict_bboxes)
