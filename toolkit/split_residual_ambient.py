"""
Created by Zhong Yuhui
"""
from tqdm import tqdm
import meshio
import numpy as np
import pyproj
import shapely
import os
import gc
from argparse import ArgumentParser

TRANSPROJ = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    # "EPSG:3857",
    pyproj.CRS("+proj=tmerc +lat_0=0 +lon_0=121.4655555555555 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"),
    always_xy=True,
)

O_LONG = 121.46708055555555556  # 东经121°28'2.21"东
O_LAT = 31.23543055555555556    # 31°14'7.55"北

OX, OY = pyproj.Transformer.from_crs(
    "EPSG:4326",
    pyproj.CRS("+proj=tmerc +lat_0=0 +lon_0=121.4655555555555 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"),
    always_xy=True,
).transform(O_LONG, O_LAT)

BUFFER_RADIUS = 2.0

ROT = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

def split_non_building_ply(data_dir, ply_path, output_prefix):

    vertices = meshio.read(ply_path)
    vertices_points = vertices.points
    vertices_point_data = vertices.point_data

    # indices = np.arange(len(vertices.points), dtype=np.int64)
    # indices = np.random.choice(indices, size=int(len(vertices.points)//8), replace=False)
    # vertices_points = vertices.points[indices]
    # vertices_point_data = dict()
    # for key in vertices.point_data:
    #     vertices_point_data[key] = vertices.point_data[key][indices]
    # vertices_point_data = vertices.point_data[indices]

    del vertices

    xyz = vertices_points

    with open(os.path.join(data_dir, "shp_merge.geojson")) as f:
        geoms = shapely.from_geojson(f.read()).geoms
    gc2loc = np.loadtxt(os.path.join(data_dir, "gc_to_local.txt"))
    align_cc = np.loadtxt(os.path.join(data_dir, "align_cc.txt"))

    loc2gc = np.linalg.inv(gc2loc)

    xyz_wgs84 = xyz @ loc2gc[:3, :3].T + loc2gc[None, :3, 3]
    xx, yy, zz = TRANSPROJ.transform(xx=xyz_wgs84[:, 0], yy=xyz_wgs84[:, 1], zz=xyz_wgs84[:, 2])
    xyz_sh2k = np.c_[xx - OX, yy - OY, zz]

    align_shp = ROT.T @ align_cc @ ROT

    del xx
    del yy
    del zz
    del xyz_wgs84
    gc.collect()

    xyz_align = xyz_sh2k @ align_shp[:3, :3].T + align_shp[:3, 3]

    pts_align = shapely.points(xyz_align[:, 0], xyz_align[:, 1])

    del xyz_align
    del xyz_sh2k
    gc.collect()

    polys_buffered = [geom.buffer(BUFFER_RADIUS) for geom in geoms]

    tree_points = shapely.STRtree(pts_align)

    pts_mask = np.ones(len(pts_align), dtype=np.bool_)

    del pts_align
    gc.collect()

    def write_single_ply(i):
        indices = tree_points.query(polys_buffered[i], predicate='contains')
        indices = indices.astype(np.int64)

        if len(indices) > 0:

            pts_mask[indices] = False

            # points = xyz[indices]
            # point_data = dict()
            # for key in vertices.point_data:
            #     point_data[key] = vertices.point_data[key][indices]

            # mesh = meshio.Mesh(points=points, point_data=point_data, cells=[])
            # meshio.write(os.path.join(output_dir, f"{i}.ply"), mesh)

    num_poly = len(polys_buffered)
    
    for _ in tqdm(map(write_single_ply, range(num_poly)), total=num_poly):
        pass

    points = xyz[pts_mask]
    point_data = dict()
    for key in vertices_point_data:
        if key != "red" and key != "green" and key != "blue":
            point_data[key] = vertices_point_data[key][pts_mask]
        else:
            point_data[key] = vertices_point_data[key][pts_mask].astype(np.uint8)

    mesh = meshio.Mesh(points=points, point_data=point_data, cells=[])
    meshio.write(output_prefix + "-ambient.ply", mesh)

    pts_mask = np.logical_not(pts_mask)

    points = xyz[pts_mask]
    point_data = dict()
    for key in vertices_point_data:
        if key != "red" and key != "green" and key != "blue":
            point_data[key] = vertices_point_data[key][pts_mask]
        else:
            point_data[key] = vertices_point_data[key][pts_mask].astype(np.uint8)

    mesh = meshio.Mesh(points=points, point_data=point_data, cells=[])
    meshio.write(output_prefix + "-residual.ply", mesh)


    # indices = tree_points.query(poly_buffered, predicate='contains')
    # indices = indices.astype(np.int64)

    # return indices

if __name__ == "__main__":
    parser = ArgumentParser(description="Split ply file to building and non-building with shapes.")
    parser.add_argument("--DATA_DIR", default=None, type=str, help="Input folder that contains 'shp_merge.geojson', 'gc_to_local.txt' and 'align_cc.txt'")
    parser.add_argument("--PLY_PATH", default=None, type=str, help="Ply file to split")
    parser.add_argument("--OUTPUT_PREFIX", default=None, type=str, help="Output ply file prefix, will output 'OUTPUT_PREFIX_building.ply' and 'OUTPUT_PREFIX_non_building.ply'")
    args = parser.parse_args()

    data_dir = args.DATA_DIR # "G:\workspace\dataset\PudongSoftwarePark_test_1km2_low500"
    ply_path = args.PLY_PATH # r"G:\workspace\dataset\PudongSoftwarePark_test_1km2_low500\gaussian_model\merge.ply"
    output_prefix = args.OUTPUT_PREFIX # r"G:\workspace\dataset\PudongSoftwarePark_test_1km2_low500\gaussian_model\building_point_cloud"

    # os.makedirs(output_dir, exist_ok=True)

    split_non_building_ply(data_dir, ply_path, output_prefix)