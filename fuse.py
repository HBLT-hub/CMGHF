import os
import re
import numpy as np

from oct1 import process_clusters
from clus import cluster_point_cloud
from VLAD import compute_vlad_for_file
from filter import filter_points

from BV_gen import process_sdf
from num import count_nearby_points
if __name__ == '__main__':
    input_file = '143_vert.txt'
    min_component_size = 100
    tolerance = 0.5
    max_neighbors = 50
    
    # 执行聚类
    clustered_points = cluster_point_cloud(
        input_file=input_file,
        min_component_size=min_component_size,
        tolerance=tolerance,
        max_neighbors=max_neighbors
    )

    filter_points(
        cluster_file='all_clusters_143_vert.txt',
        sdf_file='143_sdf.txt',
        output_file='143_sdf_new.txt',
        threshold=0.3
    )

    bvft4 = process_sdf(143)

    result = count_nearby_points('keypoints_3d_143.txt', 'all_clusters_143.txt')
    
    process_clusters(
        input_file='./all_clusters_143.txt',
        output_file="combined_angle_dist_143.txt",
        threshold=10
    )



    input_file = './combined_angle_dist_143.txt'
    vlad_feature, output_path = compute_vlad_for_file(input_file)
