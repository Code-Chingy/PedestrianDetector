from src.detection_api.base.pedestrian_detector import PedestrianDetector, Config


class DefaultConfig(Config):
    # faster_rcnn_inception_v2_pb_path = 'models/frozen_inference_graph.pb'
    model_path = 'models'
    # step_size = (6, 6)
    non_max_suppression_threshold = .2
    accuracy_threshold = .7
    img_resize = 600
    visualize = True
    verbose = True


detector = PedestrianDetector(DefaultConfig())

# img_path = 'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/' \
#            'pedestrian_detection/src/data_sets/positive/person000001.jpg'

img_path1 = 'sample_images/pedestrian.jpg'
img_path2 = 'sample_images/people.jpg'

detector.load(detector.config.model_path + '/output.svc.model')

_, results = detector.detect_with_sliding_windows(img_path1)
print(results)

# _, results = detector.detect_with_sliding_windows(img_path2)
# print(results)
#
# _, results = detector.detect_with_conv_net(img_path1)
# print(results)
#
# _, results = detector.detect_with_conv_net(img_path2)
# print(results)
