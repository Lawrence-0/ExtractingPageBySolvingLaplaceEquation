import numpy as np
from mayavi import mlab
import nrrd
from PIL import Image


print('正在计算标量场...')
X, Y, Z = np.mgrid[0:512:512j, 0:512:512j, 0:276:276j]
X1=np.asarray(X)
Y1=np.asarray(Y)
Z1=np.asarray(Z)
a1=0.000044977412412828
a2=0.275431098867723
a3=-0.226560031756573
b1=-0.0000121502464228193
b2=0.275424060774928
b3=0.00324083320501968
c1=-0.020365112085225
c2=-0.00150335752490721
c3=-0.00437751454599548
d1=9.7327859175766
d2=-0.437338645912193
d3=12.0445650985602
kx2=-0.000311793494825555
ky2=-7.22060531345529E-06
kz2=0.000319014100139011



A=(a1*np.exp(np.sqrt(-kx2)*X1)+b1*np.exp(np.sqrt(-kx2)*X1)+c1*X1+d1)
B=(a2*np.exp(np.sqrt(-ky2)*Y1)+b2*np.exp(np.sqrt(-ky2)*Y1)+c2*Y1+d2)
C=(a3*np.cos(np.sqrt(kz2)*Z1)+c3*Z1+d3)
# ellipsoid



V = A*B*C
V1=np.real(V)
print('计算完成')
########################################################################### 反转
print('正在反转标量场...')
V2=np.zeros((512,512,276))
for k in range(276):
    for j in range(512):
        for i in range(512):
            V2[511-i,j,k] = V1[i,j,k]

print('反转完成')
########################################################################### 可视化
print('正在抽取等值面...')
for k in range(276):
    for j in range(512):
        for i in range(512):
            if V2[i,j,k] >= 0.99 and V2[i,j,k]<=7.01:
                V2[i,j,k] =V2[i,j,k]
            else:
                V2[i,j,k]=0

print('等值面抽取完成')
obj=mlab.contour3d(V2,contours=7,transparent=True)
mlab.show()
############################################################################ 生成抽取矩阵
print('正在抽取第4页...')
count=0
V3=np.zeros((512,512,276))
for k in range(276):
    for j in range(512):
        for i in range(512):
            if V2[i,j,k] >= 3.9 and V2[i,j,k]<=4.1:
                V2[i,j,k] =1
                V3[i,j,k] = 65565
                count=count+1
            else:
                V2[i,j,k]=0

print('页数 4',' ','页面像素个数：', count)
print('显示拟合曲线第100层')
image3= Image.fromarray(V3[:,:,100])
image3.show()
######################################################################################### 读取nrrd
nrrd_filename = 're_0000.nrrd'
nrrd_data, nrrd_options = nrrd.read(nrrd_filename)

print('读取原始数据完成：',nrrd_options)
print('显示数据曲线第100层')
image2 = Image.fromarray(nrrd_data[:,:,100])

image2.show()
######################################################################################## 抽取页面


P4= V2 * nrrd_data

print('抽取原始页面')
#######################################################################################   保存nrrd

name='P4.nrrd'
nrrd.write(name,P4)

print('页面已保存为nrrd')

