import cv2
import time
from src.detection_api.base.pedestrian_detector import PedestrianDetector, Config


class DefaultConfig(Config):
    faster_rcnn_inception_v2_pb_path = 'models/frozen_inference_graph.pb'
    model_path = 'models'
    # step_size = (6, 6)
    non_max_suppression_threshold = .2
    accuracy_threshold = .7
    img_resize = 600
    # visualize = True
    verbose = True


detector = PedestrianDetector(DefaultConfig())

img_path1 = 'sample_images/pedestrian.jpg'
img_path2 = 'sample_images/people.jpg'

# img_path3 = '../data_sets/positive(960x720)/FudanPed00057.png'
# img_path4 = '../data_sets/positive(960x720)/FudanPed00014.png'
# img_path5 = '../data_sets/positive(960x720)/FudanPed00035.png'
# img_path6 = '../data_sets/positive(960x720)/FudanPed00048.png'

detector.load(detector.config.model_path + '/output.svc.model')

_, results = detector.detect_with_sliding_windows(img_path1)
print(results)

_, results = detector.detect_with_sliding_windows(img_path2)
print(results)

_, results = detector.detect_with_conv_net(img_path1)
print(results)

_, results = detector.detect_with_conv_net(img_path2)
print(results)

# write to image path
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#
# for path in [img_path3, img_path4, img_path5, img_path6]:
#     print(f'processing path {path}')
#     image = cv2.imread(path)
#     print(image)
#     if image is None:
#         continue
#     _, results = detector.detect_with_conv_net(image)
#     # print(results)
#     for (x, y, w, h) in results:
#         cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
#
#     writer = cv2.VideoWriter(f'{time.time()}.jpg', fourcc, 30, (image.shape[1], image.shape[0]), True)
#
#     if writer is not None:
#         writer.write(image)
#         writer.release()
#         print(f'done writing for file {path}')
