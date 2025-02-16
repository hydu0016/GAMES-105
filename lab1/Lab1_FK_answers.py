import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """
    Parses a BVH file and extracts joint names, parent indices, and offsets.

    Args:
        bvh_file_path (str): Path to the BVH file.

    Returns:
        joint_names (List[str]): Names of all joints including root and end sites.
        joint_parents (List[int]): Parent index for each joint (-1 for root).
        joint_offsets (np.ndarray): (M, 3) array containing joint offsets.
    """
    joint_names = []
    joint_parents = []
    joint_offsets = []
    stack = []
    parent_index = -1  # Root's parent index is -1

    

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        words = lines[i].strip().split()
        
        if not words:
            i += 1
            continue

        if words[0] == "ROOT" :
            joint_name = words[1]
            joint_names.append(joint_name)
            joint_parents.append(parent_index)
            #parent_index = len(joint_names) - 1
            stack.append(joint_name)

        elif words[0] == "End" or words[0] == "JOINT":
            if words[0] == "End":
                 # e.g. "End Site" => we name it "CurrentJoint_end"
                 cur_name = f"{joint_names[-1]}_end"
            else:
                 # e.g. "JOINT Spine" => child_name = "Spine"
                 cur_name = words[1]
            
            joint_names.append(cur_name)            
            joint_parents.append(joint_names.index(stack[-1]))
            stack.append(cur_name)

        elif words[0] == "OFFSET":
            offset_values = list(map(float, words[1:]))
            joint_offsets.append(offset_values)

        elif words[0] == "}":
            if stack:
                parent_index = stack.pop()

        elif words[0] == "MOTION":
            break  # Stop parsing structure after MOTION section starts

        i += 1

    joint_offsets = np.array(joint_offsets, dtype=np.float32)
    return joint_names, joint_parents, joint_offsets

def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
