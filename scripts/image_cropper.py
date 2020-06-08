# import cv2
# from skimage.feature import hog
# from src.scripts.extract_persons_annotations import rects_from_image


def scale_image_section(img, old_rect_section=None, size=None, gravity='center'):
    clone = img.copy()

    actual_image_size_y = img.shape[0]
    actual_image_size_x = img.shape[1]

    if size:
        width = size[0]
        height = size[1]
    else:
        width = actual_image_size_x
        height = actual_image_size_y

    if old_rect_section is None:
        old_rect_section = 0, 0, actual_image_size_x, actual_image_size_y

    _x, _y, _w, _h = old_rect_section
    # print(old_section)

    # cv2.rectangle(img, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), thickness=2)
    # cv2.imshow('rect', img)
    # cv2.waitKey(0)

    center_pos_x = _x + _w // 2
    center_pos_y = _y + _h // 2
    # print(center_pos_x, center_pos_y, img.shape)

    # cv2.circle(img, (center_pos_x, center_pos_y), 5, (0, 0, 0), thickness=5)
    # cv2.imshow('rect', img)
    # cv2.waitKey(0)

    new_x = center_pos_x - width // 2
    new_y = center_pos_y - height // 2
    # print(new_x, new_y)

    # cv2.line(img, (new_x, center_pos_y), (new_x + width, center_pos_y), thickness=2, color=(255, 0, 0))
    # cv2.imshow('rect', img)
    # cv2.waitKey(0)
    #
    # cv2.line(img, (center_pos_x, new_y), (center_pos_x, new_y + hieght), thickness=2, color=(255, 0, 0))
    # cv2.imshow('rect', img)
    # cv2.waitKey(0)
    #
    # cv2.circle(img, (new_x, new_y), 5, (0, 0, 0), thickness=5)
    # cv2.imshow('rect', img)
    # cv2.waitKey(0)

    # scale with center

    if gravity.lower() == 'center':
        # cv2.rectangle(img, (new_x, new_y), (new_x + width, new_y + height), (0, 255, 0), thickness=2)
        # cv2.imshow('rect', img)
        # cv2.waitKey(0)

        width = new_x + width
        height = new_y + height

    # scale to top
    if gravity.lower() == 'top':
        # cv2.rectangle(img, (new_x, _y), (new_x + width, (new_y + height) - (new_y - _y)), (0, 255, 255), thickness=2)
        # cv2.imshow('rect', img)
        # cv2.waitKey(0)

        width = new_x + width
        height = (new_y + height) - (new_y - _y)
        new_y = _y

    # scale to bottom
    if gravity.lower() == 'bottom':
        # cv2.rectangle(img, (new_x, new_y + (new_y - _y)), (new_x + width, (new_y + height) + (new_y - _y)),
        #               (255, 255, 0), thickness=2)
        # cv2.imshow('rect', img)
        # cv2.waitKey(0)

        width = new_x + width
        height = (new_y + height) + (new_y - _y)
        new_y = new_y + (new_y - _y)

    if new_x > actual_image_size_x:
        print('point extends window in x direction +')
        new_x = new_x - (width - actual_image_size_y)
    elif new_x < 0:
        print('point extends window in x direction -')
        temp = new_x
        new_x = 0
        width = width - temp

    if new_y > actual_image_size_y:
        print('point extends window in y direction +')
        new_y = actual_image_size_x - _h
        if new_y < 0:
            temp = new_y
            new_y = 0
            height = height - temp
    elif new_y < 0:
        print('point extends window in y direction -')
        temp = new_y
        new_y = 0
        height = height - temp

    return clone[new_y: height, new_x: width]




# img = cv2.imread(
#     'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/pedestrian_detection/'
#     'PennFudanPed/PNGImages/FudanPed00001.png')
# rects = rects_from_image(
#     'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/pedestrian_detection/'
#     'PennFudanPed/PNGImages/FudanPed00001.png')
#
# img1 = scale_image_section(img.copy(), None, (150, 200), gravity='center')
# img2 = scale_image_section(img.copy(), None, (150, 200), gravity='top')
# img3 = scale_image_section(img.copy(), None, (150, 200), gravity='bottom')
#
# cv2.imshow('img1', img1)
# cv2.waitKey(0)
#
# cv2.imshow('img2', img2)
# cv2.waitKey(0)
#
# cv2.imshow('img3', img3)
# cv2.waitKey(0)




# import random
#
# def _get_feature_for_image(img):
#     if img is None:
#         return
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     hist = hog(img, orientations=9, pixels_per_cell=(6, 6),
#                cells_per_block=(2, 2), block_norm='L2-Hys', visualise=False,
#                transform_sqrt=False, feature_vector=True, normalise=None)
#
#     print(type(hist), len(hist))
#
#     return hist
#
#
# def _add_features_from_image_in_dir(img):
#     # Get positive samples
#     windows = _ten_random_windows(img.copy())
#     print('group a')
#     for win in windows:
#         x, y = win
#         a=scale_image_section(img.copy(), [x,y,64,128], (64, 128))
#
#
#     print('\n\ngroup b')
#
#     for rect in rects:
#         a=scale_image_section(img.copy(), rect, (64, 128), gravity='top')
#         b=scale_image_section(img.copy(), rect, (64, 128), gravity='center')
#         c=scale_image_section(img.copy(), rect, (64, 128), gravity='bottom')
#         print(a.shape, b.shape, c.shape)
#         _get_feature_for_image(a)
#         _get_feature_for_image(b)
#         _get_feature_for_image(c)
#
#
#
# def _ten_random_windows(img):
#     h, w, _= img.shape
#     if w < 64 or h < 128:
#         return []
#
#     w = w - 64
#     h = h - 128
#
#     windows = []
#
#     for i in range(10):
#         x = random.randint(0, w)
#         y = random.randint(0, h)
#         windows.append([x, y])
#
#     return windows
#
# _add_features_from_image_in_dir(img)
