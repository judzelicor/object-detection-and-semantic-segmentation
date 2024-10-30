import numpy as np
import torch


def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the number of images from the input array
    N = images.shape[0]

    # Reshape the images
    images = images.reshape(N, 64, 64, 3) / 255.0

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = model.to(device)

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.zeros((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.zeros((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.zeros((N, 4096), dtype=np.int32)

    # add your code here to fill in pred_class and pred_bboxes
    for i in range(N):
        results = model(images[i])
        detections = results.xyxy[0]

        if len(detections) >= 2:
            # Sort detections by class for ascending order
            detections = sorted(detections, key=lambda x: int(x[5]))
            
            # Fill in pred_class and pred_bboxes for each image
            pred_class[i] = [int(detections[0][5]), int(detections[1][5])]
            pred_bboxes[i, 0] = detections[0][:4]  # x_min, y_min, x_max, y_max
            pred_bboxes[i, 1] = detections[1][:4]
        else:
            # If fewer than two detections, handle appropriately
            # For example, you could set default values:
            pred_class[i] = [-1, -1]  # Indicate missing classes
            pred_bboxes[i] = [[0, 0, 0, 0], [0, 0, 0, 0]]  # Default bbox values

        seg = np.zeros((64, 64), dtype=np.int32)
        for j in range(2):
            x_min, y_min, x_max, y_max = map(int, pred_bboxes[i, j])
            seg[y_min:y_max, x_min:x_max] = pred_class[i, j] + 1
        pred_seg[i] = seg.flatten()
        
    return pred_class, pred_bboxes, pred_seg
