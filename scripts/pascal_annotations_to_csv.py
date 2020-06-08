"""
file format

# Compatible with PASCAL Annotation Version 1.00
Image filename : "PennFudanPed/PNGImages/FudanPed00001.png"
Image size (X x Y x C) : 559 x 536 x 3
Database : "The Penn-Fudan-Pedestrian Database"
Objects with ground truth : 2 { "PASpersonWalking" "PASpersonWalking" }
# Note there may be some objects not included in the ground truth list for they are severe-occluded
# or have very small size.
# Top left pixel co-ordinates : (1, 1)
# Details for pedestrian 1 ("PASpersonWalking")
Original label for object 1 "PASpersonWalking" : "PennFudanPed"
Bounding box for object 1 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (160, 182) - (302, 431)
Pixel mask for object 1 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00001_mask.png"

# Details for pedestrian 2 ("PASpersonWalking")
Original label for object 2 "PASpersonWalking" : "PennFudanPed"
Bounding box for object 2 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (420, 171) - (535, 486)
Pixel mask for object 2 "PASpersonWalking" : "PennFudanPed/PedMasks/FudanPed00001_mask.png"

"""

import argparse
import os


def get_image_name(line):
    return (line.split('/')[-1]).replace('"', '').strip()


def get_image_size(line):
    # format = Image size (X x Y x C) : 559 x 536 x 3
    size = (line.split(':')[1]).split('x')
    return size[0].strip(), size[1].strip(), size[2].strip()


def get_entity_rect(line):
    # format Bounding box for object 2 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (420, 171) - (535, 486)
    size = (line.split(':')[1]).split('-')
    position = size[0].replace('(', '').replace(')', '').replace(' ', '').split(',')
    max_position = size[1].replace('(', '').replace(')', '').replace(' ', '').split(',')
    return position[0].strip(), position[1].strip(), max_position[0].strip(), max_position[1].strip()


def write_csv(data, csv_save_path):
    if os.path.isdir(csv_save_path):
        import time
        csv_save_path = os.path.join(csv_save_path, 'exports' + str(time.time()) + '.csv')

    lines = ""

    for details in data:
        lines += ','.join(details) + '\n'

    with open(csv_save_path, 'w') as file:
        file.write(lines)


def pascal_annotated_files_to_csv(dir_name, csv_path=None) -> list:
    if not os.path.isdir(dir_name):
        return []

    data_list = pascal_annotated_files_to_list(dir_name)

    if csv_path:
        write_csv(data_list, csv_path)

    return data_list


def pascal_annotated_files_to_list(dir_name):
    data_list: list = [
        ['filename', 'width', 'height', 'channels', 'posX', 'posY', 'maxX', 'maxY']
    ]
    file_names_list = os.listdir(dir_name)
    for file_name in file_names_list:

        file_path = os.path.join(dir_name, file_name)

        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as file:

            is_pascal_annotation = False

            width = 0
            height = 0
            channels = 0
            is_new_entity = False
            image_name = ''

            for i, line in enumerate(file):
                if i == 0:
                    # check is file is file type
                    if line.lower().__contains__('pascal annotation version'):
                        is_pascal_annotation = True
                    else:
                        break

                if is_pascal_annotation:
                    if line.lower().__contains__('image size'):
                        width, height, channels = get_image_size(line)
                    elif line.lower().__contains__('image filename'):
                        image_name = get_image_name(line)
                    elif not is_new_entity and line.lower().__contains__('details'):
                        is_new_entity = True

                    if is_new_entity and line.lower().__contains__('bounding box'):
                        pos_x, pos_y, max_x, max_y = get_entity_rect(line)
                        data_list.append([image_name, width, height, channels, pos_x, pos_y, max_x, max_y])
                        is_new_entity = False
    return data_list


def args_parser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to pascal annotations directory", )
    ap.add_argument("-o", "--output", required=True,
                    help="path to output csv file")
    ap.add_argument("-p", "--print", default=False, required=False,
                    help="displays output csv content")
    args = vars(ap.parse_args())

    return args


# dir = "C:\\Users\\Otc_Chingy\\PycharmProjects\AdvancedPython\\ai_end_of_sem_projects\\pedestrian_detection\\PennFudanPed\\Annotation"
# save_dir = "C:\\Users\\Otc_Chingy\\PycharmProjects\AdvancedPython\\ai_end_of_sem_projects\\pedestrian_detection\\PennFudanPed\\Annotation\\annotations.csv"


if __name__ == '__main__':
    # pascal_annotated_files_to_csv(dir, save_dir)
    args = args_parser()
    result = pascal_annotated_files_to_csv(args['input'], args['output'])
    if args['print']:
        for data in result:
            print(data)
