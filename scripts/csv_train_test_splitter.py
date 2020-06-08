import argparse
import csv
import os


def csv_train_test_splitter(csv_file, train_output, test_output, split_ratio=0.3, skip_column_names=False,
                            validator_func=None):
    if not os.path.exists(csv_file):
        print("csv file not found")
        return

    with open(train_output, "w") as train, open(test_output, "w") as test:
        train_writer = csv.writer(train, delimiter=",")
        test_writer = csv.writer(test, delimiter=",")
        file = open(csv_file, 'r')
        lines = file.readlines()
        reader = csv.reader(lines, delimiter=",")

        for index, row in enumerate(reader):

            if index == 0 and skip_column_names:
                continue

            if validator_func and not validator_func(row):
                print("Warning: {} contains invalid box. skipped...".format(row))
                continue

            if index <= len(lines) * split_ratio:
                test_writer.writerow(row)
            else:
                train_writer.writerow(row)

    print("\ndone!")


def args_parser():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-csv", required=True,
                    help="path to csv file", )
    ap.add_argument("-ov", "--train-output", required=True,
                    help="path to train data_sets points output csv file")
    ap.add_argument("-ot", "--test-output", required=True,
                    help="displays test data_sets points output csv content")
    ap.add_argument("-sr", "--split-ratio", required=False, default=0.3, type=float,
                    help="displays test data_sets points output csv content")
    ap.add_argument("-c", "--columns", required=False, default=None,
                    help="define what columns to keep by index or by name default is all")

    return vars(ap.parse_args())

# csv_train_test_splitter(
#     'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/pedestrian_detection/src/detection_api/csv/annotations.csv',
#     'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/pedestrian_detection/src/data_sets/full.csv',
#     'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/pedestrian_detection/src/data_sets/zero.csv',
#     0,
# )

if __name__ == "__main__":
    args = args_parser()
    csv_train_test_splitter(args['input-csv'], args['train-output'], args['test-output'], args['split-ratio'], args['rows'])
