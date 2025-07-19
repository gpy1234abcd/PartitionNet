import cv2
import numpy as np

# 读取图片
image = cv2.imread('img/000436.png')

# 定义起始点和结束点的坐标
start_point = (100, 100)
end_point = (300, 400)

# 计算箭头角度和长度
angle = np.arctan((end_point[1] - start_point[1]) / float(end_point[0] - start_point[0])) * 57.2958  # 将弧度转换为角度
length = int(np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)) // 10

# 根据角度和长度生成箭头形状
arrow_shape = np.array([[-length // 6, length * 2], [0, length / 2], [length // 6, length * 2]])
rotation_matrix = cv2.getRotationMatrix2D((length / 2, length), angle, 1)
arrow_points = rotation_matrix @ arrow_shape.T + end_point[:2].reshape(-1, 1)

# 连接起始点、箭头顶部和箭尾三个点
line_color = (0, 255, 0)  # BGR格式的线条颜色（这里设置为绿色）
cv2.polylines(image, [np.int32(list(zip(*[[start_point] + arrow_points + [(end_point)]])))], False, line_color,
              thickness=2)

# 显示结果图像
cv2.imshow("Arrow", image)
cv2.waitKey(0)
cv2.destroyAllWindows()