import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from skimage import measure
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion


with open('train4/_annotations.coco.json', 'r') as file:
    data = json.load(file)
    
def get_img(id):
    return next((item for item in data["images"] if item["id"] == id), None)

def get_img_id(id):
    image_id= get_img(id)
    return image_id["id"]

def get_img_height(id):
    img_height = get_img(id)
    return img_height["height"]

def get_img_width(id):
    img_width = get_img(id)
    return img_width["width"]
    
def mask_filled_with_0(id):
    height = get_img_height(id)
    width = get_img_width(id)
    mask = np.zeros((height,width), dtype=np.uint8)
    return mask

def get_annotations_for_img_id (id):
    return [annotation for annotation in data["annotations"] if annotation["id"] == id]

def get_img_segmentations(id):
    annotations = get_annotations_for_img_id(id)
    segmentations = []
    
    for value in annotations:
        segmentations.append(value["segmentation"][0])
    return segmentations

def binary_mask(id):
    mask = mask_filled_with_0(id)
    segmentations = get_img_segmentations(id)
    
    for value in segmentations:
        contur = np.array(value).reshape(-1,2).astype(np.int32)
        cv2.fillPoly(mask, [contur], color=1)
        
    return mask

def number_of_slice(name):
    return int(name[-7:-4])

all_ids=[item["id"] for item in data["images"]]

#sortowanie id
sorted_img = sorted(
    all_ids,
        key=lambda id: number_of_slice(get_img(id)["extra"]["name"])
) 

masks_stack=[binary_mask(slice) for slice in sorted_img]

volume_3d = np.stack(masks_stack, axis=0).astype(float)
print(volume_3d.shape)


#generowanie końców
def generate_faded_masks(slice, max_steps=10, min_voxels=20):
    
    masks = []
    current = slice.copy()

    for i in range(max_steps):
        scale = 1 - i / (max_steps + 1)  #zmniejszamy intensywność
        faded = current.astype(float) * scale
        masks.append(faded)

        current = binary_erosion(current, structure=np.ones((3, 3)))
        if np.count_nonzero(current) < min_voxels:
            break

    return masks
base_back = volume_3d[-1]
base_front = volume_3d[0]

faded_back = generate_faded_masks(base_back, max_steps=10)
faded_front = generate_faded_masks(base_front, max_steps=10)


volume_3d = np.concatenate([faded_front[::-1], volume_3d,faded_back], axis=0)

#interpolacja liniowa
pixelsize_old_x = 0.0274815
pixelsize_old_y =  0.0274815
slice_thickness_old = 0.1016

pixelsize_new_x = 0.0274815
pixelsize_new_y= 0.0274815
slice_thickness_new = slice_thickness_old/4

x_old = np.linspace(0,(volume_3d.shape[1]-1)*pixelsize_old_x, volume_3d.shape[1])
y_old = np.linspace(0,(volume_3d.shape[2]-1)*pixelsize_old_y, volume_3d.shape[2])
z_old = np.linspace(0, (volume_3d.shape[0])*slice_thickness_old, volume_3d.shape[0]) #być może tu -1, bo miedzy warstwami jest 47 przerw

method = "linear" #'cubic'
my_interpolating_object = RegularGridInterpolator((z_old,x_old, y_old), volume_3d, method=method, bounds_error=False)

x_new = np.round(volume_3d.shape[1]*pixelsize_old_x/pixelsize_new_x).astype(np.int32)
y_new = np.round(volume_3d.shape[2]*pixelsize_old_y/pixelsize_new_y).astype(np.int32)
z_new = np.arange(0, z_old[-1], slice_thickness_new)

pts = np.indices((len(z_new), x_new, y_new)).transpose((1, 2, 3, 0))
pts = pts.reshape(1, len(z_new)*x_new*y_new, 1, 3).reshape(len(z_new)*x_new*y_new, 3)
pts = np.array(pts, dtype=float)
pts[:, 1] = pts[:, 1] * pixelsize_new_x        # X
pts[:, 2] = pts[:, 2] * pixelsize_new_y        # Y
pts[:, 0] = pts[:, 0] *slice_thickness_new

print(pts.shape) #można też z tego obj wyliczyć


batch_size = 500000  # po 0.5 miliona punktów
results = []

for i in range(0, len(pts), batch_size):
    batch = pts[i:i+batch_size]
    result = my_interpolating_object(batch)
    results.append(result)

interpolated_data = np.concatenate(results)
interpolated_data = interpolated_data.reshape(len(z_new), x_new, y_new)


print(interpolated_data.shape)
print("Min:", np.min(interpolated_data))
print("Max:", np.max(interpolated_data))
voxel_count = np.count_nonzero(interpolated_data > 0)
print(f"Liczba voxelów powyżej 0: {voxel_count}")


plt.hist(interpolated_data.flatten(), bins=100)
plt.yscale("log")
plt.title("Histogram interpolated_data")
plt.xlabel("Wartości voxelów")
plt.ylabel("Liczba")
plt.show()

interpolated_data_binary = (interpolated_data > 0.1).astype(np.uint8)

# rozszerzenie maski w 3D voxele na brzegach
interpolated_data_eadges = binary_dilation(interpolated_data_binary, structure=np.ones((3, 3, 3))).astype(np.uint8)

#wiecej rozszerzeń
for _ in range(12): 
    interpolated_data_eadges = binary_dilation(interpolated_data_eadges, structure=np.ones((3,3,3)))

vertices, faces, normals, values = measure.marching_cubes(interpolated_data_eadges, 0.01, spacing=(slice_thickness_new, pixelsize_new_x, pixelsize_new_y))
# print(f"Liczba wierzchołków: {vertices.shape}")
# print(f"Liczba trójkątów: {faces.shape}")
print("X min i max:", vertices[:, 0].min(), vertices[:, 0].max())
print("Y min i max:", vertices[:, 1].min(), vertices[:, 1].max())
print("Z min i max:", vertices[:, 2].min(), vertices[:, 2].max())


faces_pv = np.hstack([[3, *face] for face in faces])
surf = pv.PolyData(vertices, faces_pv)

size=500
surf_filled = surf.fill_holes(hole_size=size)

plotter = pv.Plotter()
plotter.add_mesh(surf_filled, color="lightgray", opacity=0.7, show_edges=True)
plotter.add_blurring()
plotter.add_axes()
# plotter.add_bounding_box()
#plotter.add_measurement_widget()
plotter.show()


volume_mm3 = surf_filled.volume  #liczone przez pv
print(f"Objętość:", round(volume_mm3,2))

#liczone przez voxele samodzielnie
voxel_volume_new = pixelsize_new_x*pixelsize_new_y * slice_thickness_new
volume_mm3_new = np.count_nonzero(interpolated_data_eadges) * voxel_volume_new
print(np.count_nonzero(interpolated_data > 0.1))
print(f"Objętość guza (zliczone voxele):", round(volume_mm3_new,2))