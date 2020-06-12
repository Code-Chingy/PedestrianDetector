import cv2
from src.detection_api.base.pedestrian_detector import PedestrianDetector, Config


def write_image(image_path, save_path):

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    image = cv2.imread(image_path)

    print(image)
    if image is None:
        return

    _, results = detector.detect_with_conv_net(image)
    # print(results)
    for (x, y, w, h) in results:
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    writer = cv2.VideoWriter(save_path, fourcc, 30, (image.shape[1], image.shape[0]), True)

    if writer is not None:
        writer.write(image)
        writer.release()


# eg. write_image('images/img.jpg', 'images/new_processed_image.jpg')

class DefaultConfig(Config):
    faster_rcnn_inception_v2_pb_path = 'models/frozen_inference_graph.pb'
    model_path = 'models'
    # step_size = (6, 6)
    non_max_suppression_threshold = .2
    accuracy_threshold = .7
    img_resize = 600
    visualize = True
    verbose = True


detector = PedestrianDetector(DefaultConfig())

img_path1 = 'sample_images/pedestrian.jpg'
img_path2 = 'sample_images/people.jpg'


detector.load(detector.config.model_path + '/output.svc.model')


_, results = detector.detect_with_conv_net(img_path1)
print(results)

_, results = detector.detect_with_conv_net(img_path2)
print(results)

_, results = detector.detect_with_sliding_windows(img_path1)
print(results)

_, results = detector.detect_with_sliding_windows(img_path2)
print(results)