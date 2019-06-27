import numpy as np
a = [[1,2,3],[4,5,6]]
b = [[1,1,1],[2,2,2]]
a_1=[1,2,3]
b_1=[4,5,6]
#如果是多维列表，在使用“+”合并后，就变成了多维矩阵
#如果是一维列表，在使用“+”合并后，就变成了合并后的一维矩阵

c=a+b
c_1=a_1+b_1
print(c)
print(c_1)
#在使用extend函数时，与“+”是一样的效果
a.extend(b)
print(a)
a_1.extend(b_1)
print(a_1)

#数组的合并
a_array = np.array([[1,2,3],[4,5,6]])
b_array = np.array([[1,1,1],[2,2,2]])

#数组的纵向合并
c_array=np.vstack((a_array,b_array))
print(c_array)
c_array_1=np.vstack((a,b))
print(c_array_1)
#使用vstack合并一维列表会报错
#c_array_2=np.vstack((a_1,b_1))
#print(c_array_2)
#数组的纵向合并的另一种用法
cx=np.r_[a_array,b_array]
print(cx)

#数组的横向合并
d = np.hstack((a_array,b_array))
print(d)
d_1=np.c_[a_array,b_array]
print(d_1)