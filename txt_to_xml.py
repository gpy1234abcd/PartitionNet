# author:Hurricane
# date:  2021/4/10
# E-mail:hurri_cane@qq.com

# 将KIITI数据中的txt标注文件转换为xml格式

import os
from xml.dom.minidom import Document
from tqdm import tqdm
import cv2

# 标签文件的名子  标签的信息   图像的大小 指定获取的类别名字字典 输出的文件路径
def generate_xml(name, split_lines, img_size, class_ind, tar_dir):
	doc = Document()
	# owner
	annotation = doc.createElement('annotation')
	doc.appendChild(annotation)
	# owner
	folder = doc.createElement('folder')
	annotation.appendChild(folder)
	folder_txt = doc.createTextNode("VOC2007")
	folder.appendChild(folder_txt)

	filename = doc.createElement('filename')
	annotation.appendChild(filename)
	filename_txt = doc.createTextNode(name + '.png')
	filename.appendChild(filename_txt)
	# ones#
	source = doc.createElement('source')
	annotation.appendChild(source)

	database = doc.createElement('database')
	source.appendChild(database)
	database_txt = doc.createTextNode("The VOC2007 Database")
	database.appendChild(database_txt)

	annotation_new = doc.createElement('annotation')
	source.appendChild(annotation_new)
	annotation_new_txt = doc.createTextNode("PASCAL VOC2007")
	annotation_new.appendChild(annotation_new_txt)

	image = doc.createElement('image')
	source.appendChild(image)
	image_txt = doc.createTextNode("flickr")
	image.appendChild(image_txt)
	# onee#
	# twos#
	size = doc.createElement('size')
	annotation.appendChild(size)

	width = doc.createElement('width')
	size.appendChild(width)
	width_txt = doc.createTextNode(str(img_size[1]))
	width.appendChild(width_txt)

	height = doc.createElement('height')
	size.appendChild(height)
	height_txt = doc.createTextNode(str(img_size[0]))
	height.appendChild(height_txt)

	depth = doc.createElement('depth')
	size.appendChild(depth)
	depth_txt = doc.createTextNode(str(img_size[2]))
	depth.appendChild(depth_txt)
	# twoe#
	segmented = doc.createElement('segmented')
	annotation.appendChild(segmented)
	segmented_txt = doc.createTextNode("0")
	segmented.appendChild(segmented_txt)

	for split_line in split_lines:
		line = split_line.strip().split()
		if line[0] in class_ind:
			object_new = doc.createElement("object")
			annotation.appendChild(object_new)

			name_ = doc.createElement('name')
			object_new.appendChild(name_)
			name_txt = doc.createTextNode(line[0])
			name_.appendChild(name_txt)

			pose = doc.createElement('pose')
			object_new.appendChild(pose)
			pose_txt = doc.createTextNode("Unspecified")
			pose.appendChild(pose_txt)

			truncated = doc.createElement('truncated')
			object_new.appendChild(truncated)
			truncated_txt = doc.createTextNode("0")
			truncated.appendChild(truncated_txt)

			difficult = doc.createElement('difficult')
			object_new.appendChild(difficult)
			difficult_txt = doc.createTextNode("0")
			difficult.appendChild(difficult_txt)

			#新增加一个属性 depth 把目标离摄像机的距离也弄上
			depth=doc.createElement('depth_object')
			object_new.appendChild(depth)
			depth_txt=doc.createTextNode(str(int(float(line[-2]))))
			depth.appendChild(depth_txt)



			# threes-1#
			bndbox = doc.createElement('bndbox')
			object_new.appendChild(bndbox)

			xmin = doc.createElement('xmin')
			bndbox.appendChild(xmin)
			xmin_txt = doc.createTextNode(str(int(float(line[4]))))
			xmin.appendChild(xmin_txt)

			ymin = doc.createElement('ymin')
			bndbox.appendChild(ymin)
			ymin_txt = doc.createTextNode(str(int(float(line[5]))))
			ymin.appendChild(ymin_txt)

			xmax = doc.createElement('xmax')
			bndbox.appendChild(xmax)
			xmax_txt = doc.createTextNode(str(int(float(line[6]))))
			xmax.appendChild(xmax_txt)

			ymax = doc.createElement('ymax')
			bndbox.appendChild(ymax)
			ymax_txt = doc.createTextNode(str(int(float(line[7]))))
			ymax.appendChild(ymax_txt)

	# 将DOM对象doc写入文件
	with open(tar_dir + name + '.xml', "wb") as f:
		f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))

	f.close()


if __name__ == '__main__':
	# 指定获取的类别
	# class_ind = ('Pedestrian', 'Car', 'Truck',  'Cyclist')
	class_ind = ('Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram')

	# labels_dir    指定KIITI数据集中TXT文件路径
	# img_dir       指定KIITI数据集中图片文件路径
	# tar_dir       指定KIITI数据集标注转换为xml文件的路径
	labels_dir = './training/label_2/'
	img_dir = './data_object_image_2/training/image_2/'
	tar_dir = r'E:\\gpy\\CV_ToolBox-master\\annotation\\'

	for parent, dirnames, filenames in os.walk(labels_dir):  # 分别得到根目录，子目录和根目录下文件
		for file_name in tqdm(filenames):
			full_path = os.path.join(parent, file_name)  # 获取文件全路径
			# 打开标签文件
			f = open(full_path)
			split_lines = f.readlines()
			name = file_name[:-4]  # 后四位是扩展名.txt，只取前面的文件名
			img_path = os.path.join(img_dir, name + '.png')  # 路径需要自行修改
			# print(img_path)
			img_size = cv2.imread(img_path).shape
			# 标签文件的名子  标签的信息   图像的大小 指定获取的类别名字字典 输出的文件路径
			generate_xml(name, split_lines, img_size, class_ind, tar_dir)
print('ALL Done!')

