import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    """ 返回一个2 dimensional numpy array"""
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


    joint_num=len(joint_name)

    joint_positions = np.empty((joint_num,3))
    joint_orientations = np.empty((joint_num,4))

    joint_positions[0]=motion_data[frame_id][:3]
    joint_orientations[0]=R.from_euler('XYZ', motion_data[frame_id][3:6], degrees=True).as_quat()

    count=1

    for i in range(1,joint_num):        
        if joint_name[i].endswith('_end'):
            joint_orientations[i]=(R.from_quat(joint_orientations[joint_parent[i]])* (R.from_euler('XYZ', [0.,0.,0.], degrees=True))).as_quat()
        else:        
            joint_orientations[i]=(R.from_quat(joint_orientations[joint_parent[i]])* R.from_euler('XYZ', motion_data[frame_id][3+count*3:6+count*3], degrees=True)).as_quat()
            count+=1
            
        joint_positions[i]= joint_positions[joint_parent[i]]+np.dot(R.from_quat(joint_orientations[joint_parent[i]]).as_matrix(),joint_offset[i])
                

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

    #根据bone的index，找到motion data中对应数据的位置
    def index_bone_to_channel(index, flag):
        if flag == 't':
            end_bone_index = end_bone_index_t
        else:
            end_bone_index = end_bone_index_a
        for i in range(len(end_bone_index)):
            if end_bone_index[i] > index:
                return index - i
        return index - len(end_bone_index)
    
    #因为我们要把a pose 转变成a pose，所以需要把shoulder 后面所有的子bone都要旋转一下
    #tpose和apose的区别其实就是肩膀旋转了一个角度，a变t的过程就是把肩膀旋转回去，但是我们一定要把后面所有的bone都旋转
    #为什么所有的子bone都要旋转呢？如果你不旋转子bone的话 那么新的姿势就是子bone还是保持原来的apose但是肩膀变好了，想象一个人人体晾衣架
    def get_t2a_offset(bone_name):
        l_bone = ['lShoulder', 'lElbow', 'lWrist'] 
        r_bone = ['rShoulder', 'rElbow', 'rWrist']
        if bone_name in l_bone:
            return R.from_euler('XYZ', [0.,0.,45.], degrees=True)
        if bone_name in r_bone:
            return R.from_euler('XYZ', [0.,0.,-45.], degrees=True)
        return R.from_euler('XYZ', [0.,0.,0.], degrees=True)

    #这个也是工具，辅助第一个工具函数的
    end_bone_index_t = []
    for i in range(len(t_name)):
        if t_name[i].endswith('_end'):
            end_bone_index_t.append(i)

    end_bone_index_a = []
    for i in range(len(a_name)):
        if a_name[i].endswith('_end'):
            end_bone_index_a.append(i)



    #真正的代码从这里开始
    motion_data = load_motion_data(A_pose_bvh_path)

    t_name, t_parent, t_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_name, a_parent, a_offset = part1_calculate_T_pose(A_pose_bvh_path)



    for m_i in range(len(motion_data)):
        frame = motion_data[m_i]
        cur_frame = np.empty(frame.shape[0])
        cur_frame[:3] = frame[:3]
        for t_i in range(len(t_name)):
            cur_bone = t_name[t_i]
            a_i = a_name.index(t_name[t_i])
            if cur_bone.endswith('_end'):
                continue
            channel_t_i = index_bone_to_channel(t_i, 't')
            channel_a_i = index_bone_to_channel(a_i, 'a')
            
            # retarget
            local_rotation = frame[3+channel_a_i*3 : 6+channel_a_i*3]
            if cur_bone in ['lShoulder', 'lElbow', 'lWrist', 'rShoulder', 'rElbow', 'rWrist']:
                p_bone_name = t_name[t_parent[t_i]]
                Q_pi = get_t2a_offset(p_bone_name)
                Q_i = get_t2a_offset(cur_bone)
                local_rotation = (Q_pi * R.from_euler('XYZ', local_rotation, degrees=True) * Q_i.inv()).as_euler('XYZ', degrees=True)
            cur_frame[3+channel_t_i*3 : 6+channel_t_i*3] = local_rotation

        motion_data[m_i] = cur_frame

    return motion_data



            
            
            



    