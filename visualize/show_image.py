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


def show_image(img,
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


if __name__ == '__main__':
    for n in range(1, 181, 30):
        # 读取每一张图片
        img_folder = '/data/GOT-10k/test/GOT-10k_Test_{:06d}/'.format(n)

        predict_bboxes = []
        with open(
                "visualize/got-10k-test/Tiny/GOT-10k_Test_{:06d}/GOT-10k_Test_{:06d}_001.txt"
                .format(n, n)) as fp:
            lines = fp.readlines()
            # print(lines)
            for line in lines:
                predict_bbox = np.fromstring(line, dtype=float, sep=",")
                predict_bboxes.append(predict_bbox)

        for i in range(len(os.listdir(img_folder)) - 1):
            img_file = os.path.join(img_folder, '{:08d}.jpg'.format(i + 1))
            # print(img_file)

            # (1) 获取 image
            color_fmt = "RGB"
            img = read_image(img_file, color_fmt)
            # (2) 获取 predict_bbox
            predict_bbox = predict_bboxes[i]
            # (3) 展示添加预测框的图片
            show_image(img, predict_bbox)
