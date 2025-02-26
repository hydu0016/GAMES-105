import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
import task2_inverse_kinematics

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    #由于提供的joint_orientations是每个关节的朝向(global rotation->Q)，所以我们必须要自己计算关节的本地旋转(R)
    def get_joint_rotations():
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat() #注意这里的inv(), 由于我们需要求每个关节的本地旋转R，但是我们知道这个关节的global旋转Q和父关节Q，那么本地旋转就是父关节Q的转置×本关节的Q
        return joint_rotations

    #meta data里给了关节位置，但是没有给关节offset所以我们需要自己计算,计算方法就是这个关节的位置减去父关节的位置
    def get_joint_offsets():
        joint_offsets = np.empty(joint_positions.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_offsets[i] = np.array([0.,0.,0.])
            else:
                joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
        return joint_offsets   
     
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    #如果Path2长度是1的话，说明这个path2只有一个root节点，那么就证明fixed joint到end joint不会经过root
    if len(path2) == 1:
        path2 = []

    
    # 根据每个关节目前的global orientation（Q） 计算得到每个joint的local rotation (R)
    joint_rotations = get_joint_rotations()
    # 根据每个关节的位置计算的到关节关节之间的offset
    joint_offsets = get_joint_offsets()

    # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
    #创建两个list来存储这个关节链上所有joint的rotation和offset
    rotation_chain = np.empty((len(path), 3), dtype=float)
    offset_chain = np.empty((len(path), 3), dtype=float)

 

    # position_chain[0] = joint_positions[path[0]]
    start_position = torch.tensor(joint_positions[path[0]], requires_grad=False)
    offset_chain[0] = np.array([0.,0.,0.])


    # 对chain进行初始化
    if len(path2) > 1:
        rotation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv().as_euler('XYZ')
    else:
        rotation_chain[0] = R.from_quat(joint_orientations[path[0]]).as_euler('XYZ')



    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations