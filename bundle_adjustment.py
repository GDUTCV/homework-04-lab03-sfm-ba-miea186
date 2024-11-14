import numpy as np
import cv2


def compute_ba_residuals(parameters: np.ndarray, intrinsics: np.ndarray, num_cameras: int, points2d: np.ndarray,
                         camera_idxs: np.ndarray, points3d_idxs: np.ndarray) -> np.ndarray:
    """
    For each point2d in <points2d>, find its 3d point, reproject it back into the image and return the residual
    i.e. euclidean distance between the point2d and reprojected point.

    Args:
        parameters: list of camera parameters [r1, r2, r3, t1, t2, t3, ...] where r1, r2, r3 corresponds to the
                    Rodriguez vector. There are 6C + 3M parameters where C is the number of cameras
        intrinsics: camera intrinsics 3 x 3 array
        num_cameras: number of cameras, C
        points2d: N x 2 array of 2d points
        camera_idxs: camera_idxs[i] returns the index of the camera for points2d[i]
        points3d_idxs: points3d[points3d_idxs[i]] returns the 3d point corresponding to points2d[i]

    Returns:
        N residuals

    """
    num_camera_parameters = 6 * num_cameras # 计算相机参数的数量
    # 提取相机参数和三维点坐标
    camera_parameters = parameters[:num_camera_parameters]
    points3d = parameters[num_camera_parameters:]
    num_points3d = points3d.shape[0] // 3
    points3d = points3d.reshape(num_points3d, 3)

    # 将相机参数重塑为形状为 (num_cameras, 6) 的数组
    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    camera_rvecs = camera_parameters[:, :3] # 旋转向量
    camera_tvecs = camera_parameters[:, 3:] # 平移向量

    # 初始化外参矩阵列表
    extrinsics = []
    for rvec in camera_rvecs: # 将旋转向量转换为旋转矩阵
        rot_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics.append(rot_mtx)
    extrinsics = np.array(extrinsics)  # C x 3 x 3
    extrinsics = np.concatenate([extrinsics, camera_tvecs.reshape(-1, 3, 1)], axis=2)  # C x 3 x 4

    residuals = np.zeros(shape=points2d.shape[0], dtype=float) # 初始化残差数组
    """ 
    YOUR CODE HERE: 
    NOTE: DO NOT USE LOOPS 
    HINT: I used np.matmul; np.sum; np.sqrt; np.square, np.concatenate etc.
    """
    # 提取对应的三维点坐标
    selected_points3d = points3d[points3d_idxs]

    # 将三维点坐标齐次化，形状变为 (N, 4)
    homo_3d_points = np.concatenate((selected_points3d, np.ones((selected_points3d.shape[0], 1))), axis=1)
    homo_3d_points_T = np.transpose(homo_3d_points)

    selected_extrinsics = extrinsics[camera_idxs]  # 根据 camera_idxs 提取对应的外参矩阵
    P = np.matmul(intrinsics, selected_extrinsics)  # 计算投影矩阵，形状为 (N, 3, 4)

    # 使用投影矩阵将三维点重投影到二维平面，得到重投影点
    calculated_points2d = np.einsum('ijk,ki->ij', P, homo_3d_points_T)
    # 归一化重投影点，使其成为非齐次坐标
    calculated_points2d /= calculated_points2d[:, -1].reshape((calculated_points2d.shape[0], 1))
    calculated_points2d = calculated_points2d[:, :-1]

    # 计算观测点与重投影点之间的欧氏距离，得到残差
    residuals = np.linalg.norm(points2d - calculated_points2d, axis=1)
    """ END YOUR CODE HERE """
    return residuals
