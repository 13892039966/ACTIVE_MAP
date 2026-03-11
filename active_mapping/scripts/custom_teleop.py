#!/usr/bin/env python3
# scripts/custom_teleop.py
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from pynput.keyboard import Key, KeyCode, Listener

class CustomKeyboardTeleop:
    def __init__(self):
        rospy.init_node('custom_keyboard_teleop', anonymous=True)
        
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        
        self.linear_speed = 2.5  
        self.angular_speed = 1.0 
        
        # 🌟 状态机新增：抬头(pitch_up) 和 低头(pitch_down)
        self.active_keys = {
            'forward': False, 'backward': False,
            'left': False, 'right': False,
            'up': False, 'down': False,
            'turn_left': False, 'turn_right': False,
            'pitch_up': False, 'pitch_down': False
        }

        rospy.loginfo("🎮 自定义键盘控制 (带 Pitch) 已启动！")
        rospy.loginfo("---------------------------")
        rospy.loginfo("⬆️ ⬇️ ⬅️ ➡️ : 前/后/左/右平移")
        rospy.loginfo("PageUp / PageDn : 上升 / 下降")
        rospy.loginfo("A / D : 向左转 / 向右转 (Yaw)")
        rospy.loginfo("W / S : 抬头 / 低头 (Pitch)")
        rospy.loginfo("R : 姿态回正 (恢复初始 Yaw/Pitch/Roll)")
        rospy.loginfo("按 ESC 键退出")

    def on_press(self, key):
        if key == Key.up:          self.active_keys['forward'] = True
        elif key == Key.down:      self.active_keys['backward'] = True
        elif key == Key.left:      self.active_keys['left'] = True
        elif key == Key.right:     self.active_keys['right'] = True
        elif key == Key.page_up:   self.active_keys['up'] = True
        elif key == Key.page_down: self.active_keys['down'] = True
        elif hasattr(key, 'char'):
            if key.char == 'a':    self.active_keys['turn_left'] = True
            elif key.char == 'd':  self.active_keys['turn_right'] = True
            # 🌟 新增 W 和 S 键监听
            elif key.char == 'w':  self.active_keys['pitch_up'] = True
            elif key.char == 's':  self.active_keys['pitch_down'] = True
            elif key.char == 'r':
                reset_cmd = Twist()
                reset_cmd.angular.x = 1.0
                self.pub.publish(reset_cmd)
                rospy.loginfo("姿态回正请求已发送")

    def on_release(self, key):
        if key == Key.up:          self.active_keys['forward'] = False
        elif key == Key.down:      self.active_keys['backward'] = False
        elif key == Key.left:      self.active_keys['left'] = False
        elif key == Key.right:     self.active_keys['right'] = False
        elif key == Key.page_up:   self.active_keys['up'] = False
        elif key == Key.page_down: self.active_keys['down'] = False
        elif hasattr(key, 'char'):
            if key.char == 'a':    self.active_keys['turn_left'] = False
            elif key.char == 'd':  self.active_keys['turn_right'] = False
            # 🌟 新增 W 和 S 键释放
            elif key.char == 'w':  self.active_keys['pitch_up'] = False
            elif key.char == 's':  self.active_keys['pitch_down'] = False
        elif key == Key.esc:
            return False

    def run(self):
        listener = Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

        rate = rospy.Rate(30)
        twist = Twist()

        while not rospy.is_shutdown() and listener.running:
            twist.linear.x = (self.active_keys['forward'] - self.active_keys['backward']) * self.linear_speed
            twist.linear.y = (self.active_keys['left'] - self.active_keys['right']) * self.linear_speed
            twist.linear.z = (self.active_keys['up'] - self.active_keys['down']) * self.linear_speed
            
            twist.angular.z = (self.active_keys['turn_left'] - self.active_keys['turn_right']) * self.angular_speed
            twist.angular.y = (self.active_keys['pitch_down'] - self.active_keys['pitch_up']) * self.angular_speed

            self.pub.publish(twist)
            rate.sleep()

if __name__ == '__main__':
    try:
        CustomKeyboardTeleop().run()
    except rospy.ROSInterruptException:
        pass
