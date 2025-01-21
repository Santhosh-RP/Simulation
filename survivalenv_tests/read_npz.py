import sys

import numpy as np
import matplotlib.pyplot as plt

import cv2

x = []
y = []

# for i in range(0, 3444, 10):
#     fd = open(f'/dl/episode_{str(i).zfill(6)}.npz', 'br')
#     read = np.load(fd, allow_pickle=True)
#     l = len(read["data"])
#     x.append(i)
#     y.append(l)
#     print(i, l)
# plt.plot(x, y)
# plt.show()

fd = open(sys.argv[1], 'br')
read = np.load(fd, allow_pickle=True)

def nop(a):
    pass

cv2.namedWindow('window')
steps = len(read['data'])
cv2.createTrackbar("health", "window", 0, 2000, nop)
cv2.createTrackbar("steps", "window", 0, steps, nop)

print(read['data'][0].keys())
print(read['data'][0]['observation'].keys())
for on in read['data'][0]['observation'].keys():
    ov = read['data'][0]['observation'][on]
    print(on, ov.shape)

for stepi, step in enumerate(read['data']):
    # image
    image = step['observation']['view']
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # health
    health = step['observation']['health']
    red = int((1.-health)*255)
    green = int(health*255)

    cv2.setTrackbarPos(trackbarname="health", winname="window", pos=int(2000.*health))
    cv2.setTrackbarPos(trackbarname="steps", winname="window", pos=int(stepi))



    outimg = cv2.resize(image, (0,0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('window', outimg)
    cv2.waitKey(1)