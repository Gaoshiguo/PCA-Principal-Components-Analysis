import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
x=[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]
y=[2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]
means_X=round(float(sum(x)/(len(x))),4)
means_Y=round(float(sum(y)/(len(y))),4)
print("x 维的平均值为：",means_X)
print("y 维的平均值为：",means_Y)

update_x=[]
update_y=[]
for i in range(0,len(x)):
    var=x[i]-means_X
    update_x.append(var)
    i=i+1
print(update_x)
for i in range(0,len(x)):
    var=y[i]-means_Y
    update_y.append(var)
    i=i+1
print(update_y)
#将两个数组纵向合并
c=np.vstack((update_x,update_y))
print(c)
#调用numpy包计算协方差
cov_c=np.cov(c)
cov_st=np.array(cov_c)
print(type(cov_st))


#求这个协方差矩阵的特征值和特征向量

A,B=np.linalg.eig(cov_st)
print("该协方差矩阵的特征值为：",A)
print("该协方差矩阵的特征向量为：",B)
print(B)
vetor=[]
for i in range(0,len(B)):
    vetor.append(B[i][0])
    i=i+1

vetor=np.array(vetor)
print(vetor)

new_mertix=np.dot(vetor,c)
print(new_mertix)
# 将全局的字体设置为黑体
matplotlib.rcParams['font.family'] = 'SimHei'
plt.scatter(update_x, update_y,s=150,color='blue',marker='.')
plt.xlabel("x")#代表y坐标轴的名字
plt.ylabel("y")#代表x坐标轴的名字
plt.title("降维前的初始数据为：")
plt.show()

##使用两个特征向量
new_mertix_1 = np.dot(B,c)
print(new_mertix_1)
print(len(new_mertix_1[0]))

plt.scatter(new_mertix_1[0], new_mertix_1[1],s=150,color='blue',marker='.')
plt.xlabel("x")#代表y坐标轴的名字
plt.ylabel("y")#代表x坐标轴的名字
plt.title("降维前的初始数据为：")
plt.show()