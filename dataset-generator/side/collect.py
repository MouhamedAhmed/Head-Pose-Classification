import os
import shutil
from shutil import copy2
import re 

def main():
	folders_paths = [x[0] for x in os.walk('./Head-Pose-Annotations-Dataset/')][1:]
	i = 0
	while i < len(folders_paths):
		if len(folders_paths[i]) < 8 or folders_paths[i][-8:-2] != 'Person':
			del folders_paths[i]
			i -= 1
		i += 1


	side_images_paths = []
	for folder_path in folders_paths:
		# get all images paths
		included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
		images_paths = [fn for fn in os.listdir(folder_path)
					if any(fn.endswith(ext) for ext in included_extensions)]
		
		# get required side images paths
		for path in images_paths:
			path_split = re.split('\.|\-|\+', path)
			tilt = int(path_split[-3])
			pan = int(path_split[-2])
			if pan >= 45 and tilt <= 30:
				side_images_paths.append(folder_path + '/' + path)

		# remove and re-create dst folder
		dst_folder_path = './side_dataset'
		if os.path.exists(dst_folder_path) and os.path.isdir(dst_folder_path):
			shutil.rmtree(dst_folder_path)
		os.mkdir(dst_folder_path[2:])

	# copy images
	for src_path in side_images_paths:
		dst_path = dst_folder_path + '/' + src_path.split('/')[-1]
		copy2(src_path,dst_path)




if __name__ == '__main__':
	main()




