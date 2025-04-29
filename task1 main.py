import numpy as np
import matplotlib.pyplot as plt

# Load training and test data
data = np.load('face_alignment_training_images.npz', allow_pickle=True)
test_data = np.load('face_alignment_test_images.npz', allow_pickle=True)

# Obtain information from data
images = data['images']
pts = data['points']
test_images = test_data['images']

print(images.shape, pts.shape)  # (2811, 256, 256, 3), (2811, 5, 2)
print(test_images.shape)        # (554,  256, 256, 3)

# Useful functions taken from worksheet
def visualise_pts(img, pts):
    plt.imshow(img)
    plt.plot(pts[:, 0], pts[:, 1], '+r')
    plt.show()

def euclid_dist(pred_pts, gt_pts):
    pred_pts = np.reshape(pred_pds, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts-gt_pts), axis=-1))

def save_as_csv(points, location = "."):
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==5*2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
    np.savetxt(location + '/results_task2.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')    

for i in range(3):
    idx = np.random.randint(0, images.shape[0])
    visualise_pts(images[idx, ...], pts[idx, ...])


    

