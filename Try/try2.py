import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

hex_num = '972a19b0dc047f482972e513446d3735ea7d41aadaa57e0201a5f0f9a93fedbb'
hex_dict = {'0': '0000', '1': '0001', '2': '0010', '3': '0011', '4': '0100', '5': '0101', '6': '0110', '7': '0111', '8': '1000', '9': '1001', 'a': '1010', 'b': '1011', 'c': '1100', 'd': '1101', 'e': '1110', 'f': '1111'}
bi = ''
for digit in hex_num:
    bi += hex_dict[digit]
print(bi)
data = []
bcnt = 7
sum = 0
cnt = 0
for i, num in enumerate(bi):
    cnt += 1
    sum = sum + (int(num) * pow(2, bcnt))
    bcnt -= 1
    if cnt == 8:
        data.append(sum)
        cnt = 0
        bcnt = 7
        sum = 0
print(data)
print(len(data))

width = 32
height = len(data) // width
# if len(data) % width != 0:
#     # 填充 0 到能整除 width 的长度
#     padded_length = width * (height + 1)
#     data = np.pad(data, (0, padded_length - len(data)), 'constant', constant_values=0)
#     height += 1

image = np.array(data).reshape((height, width))
# print(image)
# print(image.shape)

plt.imshow(image, cmap='gray', aspect='auto')
plt.axis('off')
plt.show()

# import numpy
# from PIL import Image
# import binascii

# def getMatrixfrom_bin(filename,width):
#     with open(filename, 'rb') as f:
#         content = f.read()
#     hexst = binascii.hexlify(content)  #将二进制文件转换为十六进制字符串
#     print(hexst)
#     fh = numpy.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])  #按字节分割
#     rn = int(len(fh)/width)
#     fh = numpy.reshape(fh[:rn*width],(-1,width))  #根据设定的宽度生成矩阵
#     fh = numpy.uint8(fh)
#     return fh

# filename = "D:\\vscode\\mal1.txt"
# matrix = getMatrixfrom_bin(filename,256)
# print(matrix.shape)
# im = Image.fromarray(matrix) #转换为图像
# im.show()

# from PIL import Image
# from numpy import asarray
 

# # load the image and convert into 
# # numpy array
# img = Image.open("D:\\download\\002ce0d28ec990aadbbc89df457189de37d8adaadc9c084b78eb7be9a9820c81.png")
# numpydata = asarray(img)
 
# # data
# print(numpydata)
