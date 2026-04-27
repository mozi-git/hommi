
# write a script that loads all data from the directory and replace "gripperCalibrationRunName" in {left}.json and {right}.json with the gripper calibration run name from the gripper calibration json file
import os
import json

def overwrite_gripper(demonstration_iterator):
    # left_gripper_dir = '2025-08-26T19-54-42.951Z_53888_BimanualCupHandover-patio1_grippercalibration_left'
    # right_gripper_dir = '2025-08-26T19-53-54.578Z_54739_BimanualCupHandover-patio1_grippercalibration_right'
    right_gripper_dir = '2026-01-26T09-40-32.803Z_73372_waiter2_grippercalibration_right'
    left_gripper_dir = '2026-01-26T09-40-51.744Z_32688_waiter2_grippercalibration_left'
    # for gripper_dir in demonstration_iterator(['grippercalibration']):
    #     if os.path.exists(os.path.join(gripper_dir, 'left.json')):
    #         left_gripper_dir = gripper_dir
    #     if os.path.exists(os.path.join(gripper_dir, 'right.json')):
    #         right_gripper_dir = gripper_dir

    # print(left_gripper_dir, right_gripper_dir)
    # exit()
    
    for demonstration_dir in demonstration_iterator(['demonstration']):
        for side in ['left', 'right']:
            json_path = os.path.join(demonstration_dir, f'{side}.json')
            if not os.path.exists(json_path):
                continue
            
            with open(json_path, 'r') as f:
                data = json.load(f)

            if side == 'left' and left_gripper_dir:
                data['gripperCalibrationRunName'] = left_gripper_dir
            elif side == 'right' and right_gripper_dir:
                data['gripperCalibrationRunName'] = right_gripper_dir

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
