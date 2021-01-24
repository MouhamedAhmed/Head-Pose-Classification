
import os
from shutil import copy2
import shutil
import numpy as np

def main():
	current_directory = os.getcwd()
	final_directory = os.path.join(current_directory, r'back_dataset')
	if not os.path.exists(final_directory):
	   os.makedirs(final_directory)
	else:
	    shutil.rmtree(final_directory)
	    os.makedirs(final_directory)

	folders = [f for f in os.listdir('./simple_images')]
	files = []
	for folder in folders:
	    files.extend([os.path.join(current_directory , 'simple_images' + '/' + folder + '/' + f) for f in os.listdir('./simple_images/'+folder)])

	count = 0
	current_directory = os.getcwd()
	for f in files:
	    src_path = f
	    # dst_path = './back_dataset/'+str(count)+'.jpeg'
	    dst_path = os.path.join(current_directory, 'back_dataset/' + str(count)+'.jpeg')
	    count += 1
	    d = copy2(src_path, dst_path)

if __name__ == '__main__':
	main()


