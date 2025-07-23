# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import time
import os
from pure_pursuit_cuda import pure_pursuit_cuda


def vector_angle(w: np.ndarray, v: np.ndarray):
    delta_angle = np.arctan2(
        w[1] * v[0] - w[0] * v[1], 
        w[0] * v[0] + w[1] * v[1]
    )
    return delta_angle


def nearest_point_on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    a2b = b - a
    a2c = c - a
    a2b_mag = np.sqrt(np.sum(a2b**2))
    a2b_norm = a2b / (a2b_mag + 1e-6)
    dist = np.dot(a2c, a2b_norm)
    if dist < 0:
        return a, dist
    elif dist > a2b_mag:
        return b, dist
    else:
        return a + a2b_norm * dist, dist
    

class PathHelper:
    def __init__(self, points: np.ndarray):
        self.points = points
        self._init_point_distances()
        self._gpu_op = pure_pursuit_cuda(self.points,self._point_distances)
        self._debug_mode = os.environ.get('PATH_PLAN_GPU_DEBUG', '0') == '1'

    def _init_point_distances(self):
        self._point_distances = np.zeros(len(self.points))
        length = 0.
        for i in range(0, len(self.points) - 1):
            self._point_distances[i] = length
            a = self.points[i]
            b = self.points[i + 1]
            dist = np.sqrt(np.sum((a - b)**2))
            length += dist
        self._point_distances[-1] = length

    def point_distances(self):
        return self._point_distances

    def get_path_length(self):
        length = 0.
        for i in range(1, len(self.points)):
            a = self.points[i - 1]
            b = self.points[i]
            dist = np.sqrt(np.sum((a - b)**2))
            length += dist
        return length
    
    def points_x(self):
        return self.points[:, 0]
    
    def points_y(self):
        return self.points[:, 1]
    
    def get_segment_by_distance(self, distance):

        for i in range(0, len(self.points) - 1):
            d_b = self._point_distances[i + 1]

            if distance < d_b:
                return (i, i + 1)
            
        i = len(self.points) - 2

        return (i, i + 1)

    def get_segment_by_distance_and_seg_id(self, distance, seg_id):
        n = len(self.points)
        # Validate seg_id to avoid out-of-range indexing
        if seg_id < 0:
            seg_id = 0
        elif seg_id >= n - 1:
            seg_id = n - 2
        
        # If distance is less than the distance at seg_id, search backwards
        if distance < self._point_distances[seg_id]:
            # Search backwards to find the first segment where distance fits
            for i in range(seg_id, 0, -1):
                if distance >= self._point_distances[i - 1]:
                    return (i - 1, i)
            # If not found, it means distance is smaller than all points; return first segment
            return (0, 1)
        else:
            # If distance is greater than or equal to the distance at seg_id, search forwards
            for i in range(seg_id, n - 1):
                if distance < self._point_distances[i + 1]:
                    return (i, i + 1)
            # If not found, distance is beyond last point; return last segment
            return (n - 2, n - 1)

    def get_point_by_distance(self, distance,seg_id):
        a_idx, b_idx = self.get_segment_by_distance_and_seg_id(distance,seg_id)
        if self._debug_mode:
            a_idx_raw, b_idx_raw = self.get_segment_by_distance(distance)
            print(f"kanghui debug get point by distance ========> a_idx = {a_idx} a_idx_raw = {a_idx_raw} self.points shape = {self.points.shape}")
        #if a_idx != a_idx_raw: print(f"kanghui debug get point by distance ========> a_idx = {a_idx} a_idx_raw = {a_idx_raw}")


        a, b = self.points[a_idx], self.points[b_idx]
        a_dist, b_dist = self._point_distances[a_idx], self._point_distances[b_idx]
        u = (distance - a_dist) / ((b_dist - a_dist) + 1e-6)
        u = np.clip(u, 0., 1.)
        return a + u * (b - a)
    
    def find_nearest(self, point):
        if self._debug_mode:
            start = time.perf_counter()
        
        gpu_min_pt , gpu_dist_along_seg , gpu_dist_to_seg, gpu_pt_seg = self._gpu_op.find_nearest(point)
        #print(f"kanghui debug gpu_min_pt = {gpu_min_pt}")
        #print(f"kanghui debug gpu_dist_along_seg = {gpu_dist_along_seg}")
        #print(f"kanghui debug gpu_pt_seg = {gpu_pt_seg}")
        #print(f"kanghui debug self.points = {self.points}")
        if self._debug_mode:
            print(f"kanghui debug gpu op time {(time.perf_counter() - start) * 1000}")
            start = time.perf_counter()
            min_pt_dist_to_seg = 1e9
            min_pt_seg = None
            min_pt = None
            min_pt_dist_along_path = None

            for a_idx in range(0, len(self.points) - 1):
                b_idx = a_idx + 1
                a = self.points[a_idx]
                b = self.points[b_idx]
                nearest_pt, dist_along_seg = nearest_point_on_segment(a, b, point)
                dist_to_seg = np.sqrt(np.sum((point - nearest_pt)**2))

                if dist_to_seg < min_pt_dist_to_seg:
                    min_pt_seg = (a_idx, b_idx)
                    min_pt_dist_to_seg = dist_to_seg
                    min_pt = nearest_pt
                    min_pt_dist_along_path = self._point_distances[a_idx] + dist_along_seg

            
            print(f"kanghui debug cpu op time {(time.perf_counter() - start) * 1000}")
            print(f"kanghui debug min_pt ind 0 gpu = {gpu_min_pt[0]}  cpu = {min_pt[0]} diff = {gpu_min_pt[0] - min_pt[0]}")
            print(f"kanghui debug min_pt ind 1 gpu = {gpu_min_pt[1]}  cpu = {min_pt[1]} diff = {gpu_min_pt[1] - min_pt[1]}")
            print(f"kanghui debug dist_along_seg gpu = {gpu_dist_along_seg}  cpu = {min_pt_dist_along_path} diff = {gpu_dist_along_seg - min_pt_dist_along_path}")
            print(f"kanghui debug dist_to_seg gpu = {gpu_dist_to_seg}  cpu = {min_pt_dist_to_seg} diff = {gpu_dist_to_seg - min_pt_dist_to_seg}")
            if (gpu_pt_seg[0] - min_pt_seg[0]+1) != 0: print(f"kanghui debug seg id ind 0 gpu = {gpu_pt_seg[0]}  cpu = {min_pt_seg[0]} diff = {gpu_pt_seg[0] - min_pt_seg[0]}")
            if (gpu_pt_seg[1] - min_pt_seg[1]) != 0: print(f"kanghui debug seg id ind 1 gpu = {gpu_pt_seg[1]}  cpu = {min_pt_seg[1]} diff = {gpu_pt_seg[1] - min_pt_seg[1]}")
            print(f"kanghui debug seg id ind 0 gpu = {gpu_pt_seg[0]}  cpu = {min_pt_seg[0]} diff = {gpu_pt_seg[0] - min_pt_seg[0]}")
            print(f"kanghui debug seg id ind 1 gpu = {gpu_pt_seg[1]}  cpu = {min_pt_seg[1]} diff = {gpu_pt_seg[1] - min_pt_seg[1]}")
            #return min_pt, min_pt_dist_along_path, min_pt_seg, min_pt_dist_to_seg
        return gpu_min_pt , gpu_dist_along_seg , gpu_pt_seg, gpu_dist_to_seg
