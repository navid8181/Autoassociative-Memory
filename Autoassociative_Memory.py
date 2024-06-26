
# %%
from PIL import Image
import numpy as np
import os
# %%


def hardLimit(a):

    if a >= 0:
        return 1

    return -1


def getPixels(image):

    result = []

    for x in range(image.width):

        for y in range(image.height):
            pixel = image.getpixel((x, y))
            normalValue = 1 - (pixel//255)
            pixelValue = 2 * normalValue - 1
            result.append(pixelValue)

    return np.array(result)

# %%


n_imagePath = os.listdir('./data/N')
v_imagePath = os.listdir('./data/V')

n_imagesData = []
v_imagesData = []


for n in n_imagePath:
    n_image = Image.open(f'./data/N/{n}')
    n_imagesData.append(getPixels(n_image))

for v in v_imagePath:
    v_image = Image.open(f'./data/V/{v}')
    v_imagesData.append(getPixels(v_image))


# %%
print(n_imagesData)
print(v_imagesData)

# %%
#n_numpy = np.array(n_imagesData[0]).reshape(-1,1)

W =np.matrix(np.zeros((100,100)))

for n in n_imagesData:

    
    n_numpy = np.array(n).reshape(-1, 1)
    print(np.dot(n_numpy, n_numpy.reshape(1, -1)))
    W =  np.multiply(n_numpy, n_numpy.reshape(1, -1)) + W
    

for v in v_imagesData:

    v_numpy = np.array(v).reshape(-1, 1)
    W = np.multiply(v_numpy, v_numpy.reshape(1, -1)) + W

#W = np.dot(n_numpy , n_numpy.reshape(1,-1))

print(W)

# %%
hardLim = np.vectorize(hardLimit)
# %%

n_test_image = Image.open('./test/V.bmp')

n_test_numpy = getPixels(n_test_image).reshape(-1,1)

a = hardLim(np.dot(W, n_test_numpy))

print(a)

# %%
result = []
for n in n_imagesData:

    n_numpy = np.array(n).reshape(-1, 1)
 
    n_predict = n_numpy - a
   # print(np.count_nonzero(n_predict))
    #print((n_numpy == a).all())

    result.append(('ن',np.count_nonzero(n_predict)))


    if (n_numpy == a).all():
        #Zprint("ن : ", "True")
        break


for v in v_imagesData:

    v_numpy = np.array(v).reshape(-1, 1)
    v_predict = v_numpy - a
    
    #print(np.count_nonzero(v_predict))
    result.append(('و',np.count_nonzero(v_predict)))
    if (v_numpy == a).all():
        print("و :", "True")
        break

print(result)
print (f'{(sorted(result,key= lambda x : x[1])[0])[0]} : true')

# %%
