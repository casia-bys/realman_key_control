import os, sys
import time
import math
import pybullet as p
import pybullet_data
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
# ────────────────────────────── SDK / 本地模块 ──────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realman_robot.robotic_arm import *
from sensor_msgs.msg import JointState


# INTIALANGLE = [0, 0, 0, -58, 0, 0, 0]
INTIALANGLE = [0, 0, 0, -90, 0, 0, 0]
def euler_to_quaternion(euler):
    x, y, z = euler
    cy = math.cos(z * 0.5); sy = math.sin(z * 0.5)
    cp = math.cos(y * 0.5); sp = math.sin(y * 0.5)
    cr = math.cos(x * 0.5); sr = math.sin(x * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [x, y, z, w]

class RobotController:
    def __init__(self, urdf_path):
        # 连接并设置资源路径
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        cv.namedWindow("key_listener", cv.WINDOW_NORMAL)
        self.sim = True     # 修改是在仿真中还是在实际运行  !!

        # 加载机械臂 URDF，并保持基座固定
        bias_quat = euler_to_quaternion([0, 0, 0])
        self.robotId = p.loadURDF(urdf_path, [0, 0, 0], bias_quat, useFixedBase=True)
        self.numJoints = p.getNumJoints(self.robotId)

        # 初始末端位置／姿态（直接从当前仿真状态读出）
        self.current_frame_id = p.loadURDF("coordinate_frame.urdf", [0, 0, 0], useFixedBase=True)
        self.target_frame_id = p.loadURDF("coordinate_frame.urdf", [0, 0, 0], useFixedBase=True)


        if self.sim:     # 如果只用仿真
            rad_angles = [math.radians(a) for a in INTIALANGLE]
            for joint_index in range(self.numJoints):
                p.resetJointState(self.robotId, joint_index, rad_angles[joint_index])
            link_state = p.getLinkState(self.robotId, self.numJoints - 1)
            self.targetPos, self.targetOrient = link_state[0], link_state[1]
            self.jointAngles = [p.getJointState(self.robotId, i)[0] for i in range(self.numJoints)]

        else:       # 从实机中读取机器人数据
            self.realRobot = Arm(GEN72, "192.168.1.19")
            self.realRobot.Movej_Cmd(joint=INTIALANGLE, v=10, r=30)
            # self.realRobot.Movej_CANFD(joint=INTIALANGLE, follow=False, expand=0)
            print('Move to home!')
            self.jointAngles = [math.radians(angle) for angle in self.realRobot.Get_Current_Arm_State()[1]]     # 转化成弧度制 
            targetPose = self.realRobot.Get_Current_Arm_State()[2]
            self.targetPos = targetPose[0:3]
            self.targetOrient = self.calibrateEnd(0, euler_to_quaternion(targetPose[3:6]))

    def calibrateEnd(self, angle_deg, targetOrient):
        # 绕 Z 轴旋转一个固定增量（可以根据需要启用／禁用）
        angle = math.radians(angle_deg)
        rot = R.from_rotvec([0, 0, angle])
        orig = R.from_quat(targetOrient)
        return (orig * rot).as_quat()

    def translatingEnd(self, t_delta):  # 把 [±1,0,0] 类型的输入缩放到实际步长
        t_delta = [i * 0.005 for i in t_delta]
        self.targetPos = list(p.multiplyTransforms(self.targetPos, self.targetOrient, t_delta, [0, 0, 0, 1])[0])
        safe_regions = [[0.1, 1], [-0.8, 0.8], [0.1, 1.2]]        # 设置安全区域
        self.targetPos = [min(max(v, safe_regions[i][0]), safe_regions[i][1]) for i, v in enumerate(self.targetPos)]
        print(self.targetPos)
        self.targetPos = tuple(self.targetPos)      # 获取世界坐标系下的末端位置坐标xyz


    def rotatingEnd(self, axis, angle_step):
        # 计算绕 X 轴旋转的增量四元数
        q_delta = R.from_euler(axis, angle_step, degrees=True).as_quat()
        # 计算新的四元数（q_new = q_delta * q_current）
        self.targetOrient = p.multiplyTransforms(self.targetPos, self.targetOrient, [0, 0, 0], q_delta)[1]


    def applyIK(self):
        prev = self.jointAngles
        N = self.numJoints
        lower = [-math.pi]*N
        upper = [ math.pi]*N
        ranges = [math.pi]*N       # 搜索范围
        damping = [0.1]*N          # 阻尼（正则化）


        self.jointAngles = p.calculateInverseKinematics(
            bodyUniqueId         = self.robotId,
            endEffectorLinkIndex = N-1,
            targetPosition       = self.targetPos,
            targetOrientation    = self.targetOrient,
            lowerLimits          = lower,
            upperLimits          = upper,
            jointRanges          = ranges,
            restPoses            = prev,     # 偏好上一帧
            jointDamping         = damping,
            residualThreshold    = 1e-12,
            maxNumIterations     = 200
        )
        # new = self.jointAngles
        # print("Δ:", [math.degrees(n-p) for n,p in zip(new, prev)])
        
    def getPose(self):
        link_state = p.getLinkState(self.robotId, 6)
        position, orientation = link_state[0], link_state[1]
        return position, orientation

    def updateJoint(self):      # 将计算好的关节角度下发到仿真
        for i in range(self.numJoints):     
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, self.jointAngles[i])
        
        position, orientation = self.getPose()
        p.resetBasePositionAndOrientation(self.current_frame_id, position, orientation)
        p.resetBasePositionAndOrientation(self.target_frame_id, self.targetPos, self.targetOrient)


    def sendJoint(self):
        for i in range(self.numJoints):     
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, self.jointAngles[i])
        
        position, orientation = self.getPose()
        p.resetBasePositionAndOrientation(self.current_frame_id, position, orientation)
        p.resetBasePositionAndOrientation(self.target_frame_id, self.targetPos, self.targetOrient)
        # 就是一个普通的列表
        action_joint = [math.degrees(jointAngle) for jointAngle in self.jointAngles]        
        self.realRobot.Movej_CANFD(joint=action_joint,follow=False,expand=0)
        state_Joint = self.realRobot.Get_Current_Arm_State()[1]      # 这里也是一个列表 
        # ========== ① 指令关节角 ==========
        msg_cmd = JointState()
        msg_cmd.header.stamp = self.ros_node.get_clock().now().to_msg()
        msg_cmd.name    = [f'joint_{i}' for i in range(len(action_joint))]
        msg_cmd.position= action_joint
        self.pub_action.publish(msg_cmd)

        # ========== ② 实测关节角 ==========
        msg_state = JointState()
        msg_state.header.stamp = msg_cmd.header.stamp   # 相同时间戳
        msg_state.name     = msg_cmd.name
        msg_state.position = state_Joint
        self.pub_state.publish(msg_state)


    def step(self):
        # 读取键盘并映射到平移/旋转
        blank = np.zeros((10,10,3), np.uint8)
        cv.imshow("key_listener", blank)
        key = cv.waitKey(1)
        if key == ord('w'):
            self.translatingEnd([1,0,0])     
        elif key == ord('s'):
            self.translatingEnd([-1,0,0])       
        elif key == ord('a'):
            self.translatingEnd([0,1,0])       
        elif key == ord('d'):
            self.translatingEnd([0,-1,0])       
        elif key == ord('q'):
            self.translatingEnd([0,0,1])       
        elif key == ord('z'):
            self.translatingEnd([0,0,-1])       

        elif key == ord('u'):
            self.rotatingEnd('x',  1)
        elif key == ord('o'):
            self.rotatingEnd('x', -1)
        elif key == ord('i'):
            self.rotatingEnd('y',  1)
        elif key == ord('k'):
            self.rotatingEnd('y', -1)
        elif key == ord('j'):
            self.rotatingEnd('z',  1)
        elif key == ord('l'):
            self.rotatingEnd('z', -1)

        elif key == ord('r'):      # 重置
            self.realRobot.Movej_Cmd(joint=INTIALANGLE, v=10, r=30)

        # 依次执行 IK → 下发关节
        self.applyIK()
        if self.sim: 
            self.updateJoint()
        else:
            self.sendJoint()
def main():
    # 将下面路径改为你的 URDF 文件所在位置
    print("Hello ROS2  - keyboard control node running...")
    urdf = "rm_description/urdf/rm_gen72.urdf"
    controller = RobotController(urdf)
    while True:
        p.stepSimulation()
        controller.step()
        time.sleep(1/240)


if __name__ == "__main__":
    main()
