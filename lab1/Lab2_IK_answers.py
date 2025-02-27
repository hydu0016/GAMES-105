import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
import task2_inverse_kinematics

# def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
#     """
#     完成函数，计算逆运动学
#     输入: 
#         meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
#         joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
#         joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
#         target_pose: 目标位置，是一个numpy数组，shape为(3,)
#     输出:
#         经过IK后的姿态
#         joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
#         joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
#     """

#     #由于提供的joint_orientations是每个关节的朝向(global rotation->Q)，所以我们必须要自己计算关节的本地旋转(R)
#     def get_joint_rotations():
#         joint_rotations = np.empty(joint_orientations.shape)
#         for i in range(len(joint_name)):
#             if joint_parent[i] == -1:
#                 joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
#             else:
#                 joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat() #注意这里的inv(), 由于我们需要求每个关节的本地旋转R，但是我们知道这个关节的global旋转Q和父关节Q，那么本地旋转就是父关节Q的转置×本关节的Q
#         return joint_rotations

#     #meta data里给了关节位置，但是没有给关节offset所以我们需要自己计算,计算方法就是这个关节的位置减去父关节的位置
#     def get_joint_offsets():
#         joint_offsets = np.empty(joint_positions.shape)
#         for i in range(len(joint_name)):
#             if joint_parent[i] == -1:
#                 joint_offsets[i] = np.array([0.,0.,0.])
#             else:
#                 joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
#         return joint_offsets   
     
#     joint_name = meta_data.joint_name
#     joint_parent = meta_data.joint_parent
#     joint_initial_position = meta_data.joint_initial_position
#     root_joint = meta_data.root_joint
#     end_joint = meta_data.end_joint

#     path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
#     #如果Path2长度是1的话，说明这个path2只有一个root节点，那么就证明fixed joint到end joint不会经过root
#     if len(path2) == 1:
#         path2 = []

    
#     # 根据每个关节目前的global orientation（Q） 计算得到每个joint的local rotation (R)
#     joint_rotations = get_joint_rotations()
#     # 根据每个关节的位置计算的到关节关节之间的offset
#     joint_offsets = get_joint_offsets()

#     # chain和path中的joint相对应，chain[0]代表不动点，chain[-1]代表end节点
#     #创建两个list来存储这个关节链上所有joint的rotation和offset
#     rotation_chain = np.empty((len(path), 3), dtype=float)
#     offset_chain = np.empty((len(path), 3), dtype=float)

 

#     # position_chain[0] = joint_positions[path[0]]
#     start_position = torch.tensor(joint_positions[path[0]], requires_grad=False)
#     offset_chain[0] = np.array([0.,0.,0.])


#     # 对chain进行初始化
#     if len(path2) > 1:
#         rotation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv().as_euler('XYZ')
#     else:
#         rotation_chain[0] = R.from_quat(joint_orientations[path[0]]).as_euler('XYZ')



#     return joint_positions, joint_orientations



"https://github.com/DIOYF/GamesPersonalHW/blob/main/games105/Lab1_IK.py"
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


    #拿到Parent
    joint_parent = meta_data.joint_parent

    #拿到offset
    joint_offset = [meta_data.joint_initial_position[i] - meta_data.joint_initial_position[joint_parent[i]] for i in
                    range(len(joint_positions))]
    joint_offset[0] = np.array([0., 0., 0.])    

    #拿到IK Path
    joint_ik_path, _, _, _ = meta_data.get_path_from_root_to_end()
    local_rotation = []

    #拿到所有关节的local rotation
    #Now we have all joint's Q, we want get R for all joint. So Iterate through each joint index,Get all local rotation for all joints
    for i in range(len(joint_orientations)):
        # Get the parent joint index
        parent_index = joint_parent[i]

        # Convert global orientations to scipy Rotation objects
        parent_rotation = R.from_quat(joint_orientations[parent_index])
        current_rotation = R.from_quat(joint_orientations[i])

        # Compute the local rotation by applying the inverse of the parent's rotation
        local_rot = parent_rotation.inv() * current_rotation

        # Store the result in the list
        local_rotation.append(local_rot)
    local_rotation[0] = R.from_quat(joint_orientations[0])



    #  Convert our offsets, positions, and rotations into torch tensors.
    # the joint_offset_t will look like: [tensor([0., 0., 0.]), tensor([1., 2., 3.]), ...]
    joint_offset_t = [torch.tensor(data) for data in joint_offset]
    # the joint_positionT will look like: [tensor([0., 0., 0.]), tensor([1., 2., 3.]), ...]
    joint_positions_t = [torch.tensor(data) for data in joint_positions]
    # the joint_positionT will look like: [tensor([rotation matrix],requires_grad=True), tensor([1., 2., 3.]), ...]
    joint_orientations_t = [torch.tensor(R.from_quat(data).as_matrix(), requires_grad=True) for data in joint_orientations]
    #Just like above
    local_rotation_t = [torch.tensor(data.as_matrix(),requires_grad=True) for data in local_rotation]
    target_pose_t = torch.tensor(target_pose)

    epoch=300
    alpha=0.001
    for _ in range(epoch):
        for j in range(len(joint_ik_path)):
            #更新Chain上的节点的位置
            a=chain_current=joint_ik_path[j]   # Current joint in the chain.
            b=chain_parent=joint_ik_path[j-1]  # The previous joint in the chain (its "parent" in the chain).
            if j==0:
                # For the first joint in the chain, no update is necessary, as fixed joint is not move and rotate, just keep same.
                local_rotation_t[a]= local_rotation_t[a] 
                joint_positions_t[a] = joint_positions_t[a] 

            elif joint_parent[a]==b: 
                #当前节点是前一节点的子节点，说明这个chain正向

                joint_orientations_t[a]=joint_orientations_t[b] @ local_rotation_t[a]   #正向的时候，计算子关节的Q。   Q_child= Q_parent * R_child                   
                joint_positions_t[a] = joint_positions_t[b] + joint_offset_t[a] @ torch.transpose(joint_orientations_t[b],0,1)    #正向的时候，计算子关节的位置。  Pos_child= Pos_parent+ Q_parent * offset_child
                #这里为什么是torch.transpos而且旋转矩阵相乘在右边? 复制上面这行代码 搜noition
                                
            else: 
                #a=joint_parent[b] 如果当前节点是前一节点的父节点：说明现在是chain的逆向部分，比如chain是fixe_joint是foot然后end_effector是手，那么foot到root这部分在Chain上就是逆向的
                joint_orientations[a]=joint_orientations_t[b] @ torch.transpose(local_rotation_t[b],0,1)   # a是b的parent.  由于 Q_b=Q_a * R_b  =》  Q_b * R_b(转置) = Q_a * R_b * R_b(转置) =》  Qa= Q_b * R_b(转置)
                joint_positions_t[a]=joint_positions_t[b]+(-joint_offset_t[a])@torch.transpose(joint_orientations_t[a],0,1)  # a是b的parent  Pos_b= Pos_a+ Q_a * offset_b =》 Pos_a=Pos_b - Q_a * offset_b
        
        # Calculate the error: the distance between the current end joint position and the target.
        optimize_target = torch.norm(joint_positions_t[joint_ik_path[-1]] - target_pose_t)
        # Backpropagate the error to compute gradients for the rotations.
        optimize_target.backward()

        # Adjust each joint's local rotation slightly in the direction that reduces the error.
        for num in joint_ik_path:
            if local_rotation_t[num].grad is not None:
                tmp = local_rotation_t[num] - alpha * local_rotation_t[num].grad
                # Reset the tensor with the updated rotation and re-enable gradient tracking.
                local_rotation_t[num] = torch.tensor(tmp, requires_grad=True)



    # Step 5: After tweaking the chain, convert the updated rotations and positions back to the original format.
    for j in range(len(joint_ik_path)):
        a = joint_ik_path[j]
        b = joint_ik_path[j - 1]
        if j == 0:
             local_rotation[a] = R.from_matrix(local_rotation_t[a].detach().numpy())
             joint_positions[a] = joint_positions[a]
        elif b == joint_parent[a]:
             joint_orientations[a] = (R.from_quat(joint_orientations[b]) * 
                                      R.from_matrix(local_rotation_t[a].detach().numpy())).as_quat()
             joint_positions[a] = joint_positions[b] + joint_offset[a] * np.asmatrix(R.from_quat(joint_orientations[b]).as_matrix()).transpose()
        else:
             joint_orientations[a] = (R.from_quat(joint_orientations[b]) * 
                                      R.from_matrix(local_rotation_t[b].detach().numpy()).inv()).as_quat()
             joint_positions[a] = joint_positions[b] + (-joint_offset[b]) * np.asmatrix(R.from_quat(joint_orientations[a]).as_matrix()).transpose()

    # Finally, update the joints that are not in the main IK chain to ensure the whole skeleton moves consistently.
    ik_path_set = set(joint_ik_path)
    for i in range(len(joint_positions)):
        if i in ik_path_set:
            joint_orientations[i] = R.from_matrix(joint_orientations_t[i].detach().numpy()).as_quat()
        else:
            joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * local_rotation[i]).as_quat()
            joint_positions[i] = joint_positions[joint_parent[i]] + joint_offset[i] * np.asmatrix(
                R.from_quat(joint_orientations[joint_parent[i]]).as_matrix()).transpose()

    # The updated joint positions and orientations now reflect the changes required to move the end joint toward the target.
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