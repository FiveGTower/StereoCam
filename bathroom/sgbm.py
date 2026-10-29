import os
import cv2
import numpy as np

# Put this script next to left_up.png, right_up.png, left_down.png.
ROOT = os.path.dirname(os.path.abspath(__file__))
SCALE = 0.5
MAX_DISP = 96  # must be divisible by 16, at resized resolution

names = ['left_up.png', 'right_up.png', 'left_down.png']
images = []
for name in names:
    img = cv2.imread(os.path.join(ROOT, name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(name)
    images.append(cv2.resize(img, None, fx=SCALE, fy=SCALE,
                             interpolation=cv2.INTER_AREA))
left, right, down = images
if not (left.shape == right.shape == down.shape):
    raise ValueError('Input image sizes differ')

def sgbm(a, b):
    bs = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=MAX_DISP, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs,
        disp12MaxDiff=1, uniquenessRatio=7,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return stereo.compute(a, b).astype(np.float32) / 16.0

# Needs parallel cameras, and corresponding points in the right/down
# images must move towards smaller x/y than in the reference image.
dh = sgbm(left, right)
dv = sgbm(np.ascontiguousarray(left.T),
          np.ascontiguousarray(down.T)).T

vh = (dh > 0.5) & (dh < MAX_DISP - 1)
vv = (dv > 0.5) & (dv < MAX_DISP - 1)
agree = vh & vv & (np.abs(dh - dv) < np.maximum(2.5, 0.20 * np.maximum(dh, dv)))
only_h = vh & ~vv
only_v = vv & ~vh

disp = np.full(dh.shape, np.nan, np.float32)
disp[agree] = (dh[agree] + dv[agree]) / 2
# If only one pair succeeds, keep that estimate.
disp[only_h] = dh[only_h]
disp[only_v] = dv[only_v]
np.save(os.path.join(ROOT, 'disparity_trinocular.npy'), disp)

# Visualization only: fill holes and bilateral smooth, but never use
# filled disparity for metric evaluation.
filled = cv2.medianBlur(np.nan_to_num(disp, nan=0).astype(np.float32), 5)
preview = np.where(np.isfinite(disp), disp, filled)
preview = cv2.bilateralFilter(preview.astype(np.float32), 7, 12, 8)
# Fixed range: near = warm, far = cool.
scaled = (255 * np.clip((preview - 2) / 63, 0, 1)).astype(np.uint8)
color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
cv2.imwrite(os.path.join(ROOT, 'depth_preview.png'), color)
print('Saved depth_preview.png and disparity_trinocular.npy')
print('Valid disparity:', round(np.isfinite(disp).mean() * 100, 1), '%')
print('Note: this is disparity visualization, not metric depth in meters.')
