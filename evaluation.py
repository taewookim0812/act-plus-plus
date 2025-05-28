import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
import torch
import numpy as np
import cv2
from einops import rearrange
from cv_bridge import CvBridge
import os
import re
from torchvision import transforms
from policy import ACTPolicy
import pickle
from pathlib import Path
from omegaconf import OmegaConf

root_dir = Path(__file__).resolve().parent.parent

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(args_override=policy_config)
    else:
        raise NotImplementedError(f"Policy class {policy_class} is not implemented.")
    return policy

def get_image(input_image):
    curr_images = []
    curr_image = rearrange(input_image, 'h w c -> c h w')
    curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def find_best_latest_checkpoint(ckpt_root):
    best_ckpt_path = os.path.join(ckpt_root, "policy_best.ckpt")
    if os.path.isfile(best_ckpt_path):
        print(f"[INFO] Found best checkpoint: {best_ckpt_path}")
        return best_ckpt_path

    # Regex pattern to extract step number
    pattern = re.compile(r"policy_step_(\d+)_seed_\d+\.ckpt")

    best_step = -1
    best_file = None

    for fname in os.listdir(ckpt_root):
        match = pattern.match(fname)
        if match:
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_file = fname

    if best_file:
        best_ckpt_path = os.path.join(ckpt_root, best_file)
        print(f"[INFO] Found latest checkpoint by step: {best_ckpt_path}")
        return best_ckpt_path

    print("[WARN] No checkpoint found in the given directory.")
    return None


class PolicyExecutor(Node):
    def __init__(self, args):
        super().__init__('policy_executor')
        self.policy_class = args['policy_class']
        self.policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['head_camera'],
            'vq': args['use_vq'],
            'vq_class': args['vq_class'],
            'vq_dim': args['vq_dim'],
            'action_dim': 52,
            'no_encoder': args['no_encoder'],
            'temporal_agg': False
        }
        topic_cfg_path = os.path.join(root_dir, 'config', 'topic.yaml')
        config = OmegaConf.load(topic_cfg_path)
        self.topic_cfg = config.real if config.real_robot else config.sim
        self.policy = make_policy(self.policy_class, self.policy_config)
        ckpt_root = os.path.join(root_dir, args['ckpt_dir'], args['task_name'])
        best_policy_name = find_best_latest_checkpoint(ckpt_root)
        ckpt_path = os.path.join(ckpt_root, best_policy_name)

        self.policy.load_state_dict(torch.load(ckpt_path))
        self.policy.eval()
        self.policy.cuda()
        self.get_logger().info(f"Loaded policy from {ckpt_path}")

        stats_path = os.path.join(ckpt_root, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        self.query_frequency = self.policy_config['num_queries']

        self.target_joints = [f'left_joint_{i}' for i in range(1, 8)]
        self.target_joints += [f'right_joint_{i}' for i in range(1, 8)]
        self.target_joints += [f'left_hand_joint_{i}' for i in range(16)]
        self.target_joints += [f'right_hand_joint_{i}' for i in range(16)]

        self.image_subscription = self.create_subscription(
            Image, self.topic_cfg.image, self.image_callback, 10)

        self.joint_subscription_main = self.create_subscription(
            JointState, self.topic_cfg.state, self.joint_callback_main, 10)
        self.joint_subscription_left = self.create_subscription(
            JointState, '/allegroHand_left/joint_states', self.joint_callback_left, 10)
        self.joint_subscription_right = self.create_subscription(
            JointState, '/allegroHand_right/joint_states', self.joint_callback_right, 10)

        self.pedal_subscription = self.create_subscription(
            String, '/pedal_status', self.pedal_callback, 10)

        self.joint_publisher = self.create_publisher(JointState, self.topic_cfg.action, 10)

        self.ordered_action_names = self.topic_cfg.action_names

        self.bridge = CvBridge()
        self.latest_qpos = None
        self.latest_header = None
        self.joint_states_init = None

        self.latest_joint_data = {
            'joint_states': {},
            'allegroHand_left/joint_states': {},
            'allegroHand_right/joint_states': {},
        }

        self.iteration_count = 0
        self.all_actions = []

        self.clutch_on = None

    def pedal_callback(self, msg):
        if msg.data == 'Pedal Pressed':
            self.clutch_on = True
        elif msg.data == 'Pedal Released':
            self.clutch_on = False
        else:
            self.get_logger().warn("⚠️ Robot won't move — pedal switch not connected!")

    def joint_callback_main(self, msg):
        self.latest_header = msg.header
        name_to_pos = dict(zip(msg.name, msg.position))
        self.latest_joint_data['joint_states'].update(name_to_pos)

        if self.joint_states_init is None:
            joint_names_order = [
                'ewellix_lift_top_joint', 'ewellix_lift_top2_joint', 'pan_joint', 'tilt_joint',
                'left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4', 'left_joint_5',
                'left_joint_6', 'left_joint_7', 'right_joint_1', 'right_joint_2', 'right_joint_3',
                'right_joint_4', 'right_joint_5', 'right_joint_6', 'right_joint_7'
            ]
            if all(j in name_to_pos for j in joint_names_order):
                self.joint_states_init = np.array([name_to_pos[j] for j in joint_names_order], dtype=np.float32)

        self.update_latest_qpos()

    def joint_callback_left(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        self.latest_joint_data['allegroHand_left/joint_states'].update(name_to_pos)
        self.update_latest_qpos()

    def joint_callback_right(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        self.latest_joint_data['allegroHand_right/joint_states'].update(name_to_pos)
        self.update_latest_qpos()

    def update_latest_qpos(self):
        joint_map = []
        joint_map += [(f'ewellix_lift_top_joint', 'joint_states')]
        joint_map += [(f'ewellix_lift_top2_joint', 'joint_states')]
        joint_map += [(f'pan_joint', 'joint_states')]
        joint_map += [(f'tilt_joint', 'joint_states')]
        joint_map += [(f'left_joint_{i}', 'joint_states') for i in range(1, 8)]
        joint_map += [(f'right_joint_{i}', 'joint_states') for i in range(1, 8)]
        if 'real' == self.topic_cfg.mode:
            joint_map += [(f'joint_{i}.0', 'allegroHand_left/joint_states') for i in range(16)]
            joint_map += [(f'joint_{i}.0', 'allegroHand_right/joint_states') for i in range(16)]
        elif 'sim' == self.topic_cfg.mode:
            joint_map += [(f'left_hand_joint_{i}', 'joint_states') for i in range(16)]
            joint_map += [(f'right_hand_joint_{i}', 'joint_states') for i in range(16)]

        qpos = []
        for name, topic in joint_map:
            val = self.latest_joint_data[topic].get(name)
            if val is None:
                return
            qpos.append(val)

        self.latest_qpos = np.array(qpos, dtype=np.float32)

    def image_callback(self, msg):
        if self.latest_qpos is None or self.latest_header is None or self.clutch_on is False or self.clutch_on is None:
            self.get_logger().warn(f"Cannot move due to qpos: {self.latest_qpos}, header: {self.latest_header}, clutch: {self.clutch_on}")
            return

        image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        image_resized = cv2.resize(image_np, (640, 480), interpolation=cv2.INTER_AREA)
        image_tensor = get_image(image_resized)
        qpos = self.pre_process(self.latest_qpos)
        qpos_tensor = torch.tensor(qpos, dtype=torch.float32).unsqueeze(0).cuda()

        with torch.no_grad():
            if self.iteration_count % self.query_frequency == 0:
                self.all_actions = self.policy(qpos_tensor, image_tensor)
            else:
                raw_action = self.all_actions[:, self.iteration_count % self.query_frequency]
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                self.publish_joint_command(action)

            self.iteration_count += 1

    def publish_joint_command(self, raw_action):
        # print(len(raw_action), raw_action, self.clutch_on)
        if self.joint_states_init is None or len(raw_action) != 52:
            self.get_logger().warn("Invalid joint_states_init or raw_action length.")
            return

        try:
            joint_msg = JointState()
            joint_msg.header = self.latest_header
            joint_msg.name = self.ordered_action_names
            print(joint_msg.name)

            base_pos = (
                [float(self.joint_states_init[0] + self.joint_states_init[1]),
                 float(self.joint_states_init[2]),
                 float(self.joint_states_init[3])]
                if self.topic_cfg.mode == 'real'
                else list(map(float, self.joint_states_init[:4]))
            )

            offset = 4  # 0 (w/o lift & pan-tilt) or 4 (w/o lift & pan-tilt action)
            arm_pos = [float(x) for x in raw_action[offset:14+offset]]
            hand_pos = [float(x) for x in raw_action[14+offset:-2]]     # dummy action

            joint_msg.position = base_pos + arm_pos + hand_pos

            self.joint_publisher.publish(joint_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish joint command: {e}")


def main(args):
    rclpy.init()
    executor = PolicyExecutor(args)
    try:
        rclpy.spin(executor)
    except KeyboardInterrupt:
        executor.get_logger().info("Policy executor shutting down.")
    finally:
        executor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 필수 인자
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--policy_class', type=str, required=True, help='Policy class to use')
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--num_steps', type=int, required=True, help='Number of training steps')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')

    # ACT 관련 인자
    parser.add_argument('--kl_weight', type=int, default=10, help='KL Weight')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--dim_feedforward', type=int, default=3200, help='Feedforward dimension')
    parser.add_argument('--temporal_agg', action='store_true', help='Use temporal aggregation')
    parser.add_argument('--use_vq', action='store_true', help='Use vector quantization')
    parser.add_argument('--vq_class', type=int, default=None, help='Vector quantization class')
    parser.add_argument('--vq_dim', type=int, default=None, help='Vector quantization dimension')
    parser.add_argument('--no_encoder', action='store_true', help='Disable encoder')

    # 추가 설정
    parser.add_argument('--load_pretrain', action='store_true', help='Load pretrained model')
    parser.add_argument('--eval_every', type=int, default=500, help='Evaluation frequency')
    parser.add_argument('--validate_every', type=int, default=500, help='Validation frequency')
    parser.add_argument('--save_every', type=int, default=500, help='Checkpoint save frequency')

    # 인자 파싱 후 main() 실행
    main(vars(parser.parse_args()))