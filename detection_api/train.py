import numpy as np
from src.detection_api.base.pedestrian_detector import PedestrianDetector, Config, LinearSVCModel


class DefaultConfig(Config):
    model_path = 'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/' \
                 'pedestrian_detection/src/detection_api/models'
    verbose = True


detector = PedestrianDetector(DefaultConfig())
detector.set_class_dict({
    'positive': {
        'path': 'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/'
                'pedestrian_detection/src/data_sets/positive(960x720)',
        'class': 1
    },
    'negative': {
        'path': 'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/'
                'pedestrian_detection/src/data_sets/negative(720x960)',
        'class': -1,
    }
})

# save at config.model path if declared
samples = np.load(detector.config.model_path + '/samples.npy')
labels = np.load(detector.config.model_path + '/labels.npy')

svm = detector.construct(samples, labels,
                         base_model=LinearSVCModel(),
                         dev_save_name='output.dev',
                         prod_save_name='output.svc')

