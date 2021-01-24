import csv
import os

def generate_csv(dataset_path, filePath = './labels.csv'):
    # remove the csv file if exists
    if os.path.exists(filePath):
        os.remove(filePath)

    labels_folders = os.listdir(dataset_path)
    label = 0
    for folder in labels_folders:
        images = os.listdir(os.path.join(dataset_path, folder))
        for image in images:
            path = os.path.join(dataset_path, folder, image)
            
            with open(filePath, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([path, label])
        label += 1

