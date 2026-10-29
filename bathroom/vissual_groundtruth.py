import OpenEXR
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Đọc Ground Truth
exr = OpenEXR.File("groundtruth.exr")
depth = exr.channels()["Depth.V"].pixels.astype(np.float32)

# Loại bỏ giá trị không hợp lệ và nền quá xa
valid = np.isfinite(depth) & (depth > 0) & (depth < 100)

# Tự động chọn khoảng depth để tăng tương phản
z_min, z_max = np.percentile(depth[valid], [2, 98])

print(f"Depth range: {z_min:.3f} - {z_max:.3f} m")

# Vật gần sáng, vật xa tối
gray = np.zeros(depth.shape, dtype=np.uint8)

gray[valid] = np.clip(
    255 * (z_max - depth[valid]) / max(z_max - z_min, 1e-6),
    0, 255
).astype(np.uint8)

# Lưu ảnh trắng đen
cv2.imwrite("groundtruth_gray_clear.png", gray)

# Hiển thị
plt.figure(figsize=(12, 8))
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()