"""Given a folder of processed demonstrations, generates a session folder containing the demonstrations you want to use to train a policy."""

import os
import shutil
import re
from glob import glob
import hydra
from omegaconf import DictConfig

from hommi.common.generic_util import symlink_absolute
from hommi.demonstration_processing.utils.generic_util import get_demonstration_sides_present, get_demonstration_json_data
from utils.generic_util import get_demonstration_path

@hydra.main(version_base="1.2", config_path="config", config_name="create_session_iphone")
def main(cfg: DictConfig):
    # Make sure demonstration dir exists
    demonstrations_dir = cfg.demonstrations_dir
    assert os.path.isdir(demonstrations_dir)

    # Create a session folder
    session_dir = os.path.join(cfg.sessions_dir, cfg.output_session_name)
    if os.path.exists(session_dir):
        if cfg.overwrite:
            print(f'Overwriting existing session at {session_dir}')
            shutil.rmtree(session_dir)
        else:
            print(f'Session already exists at {session_dir}')
            exit()
    os.makedirs(session_dir, exist_ok=True)

    # demos
    demos_dir = os.path.join(session_dir, 'demos')
    os.makedirs(demos_dir, exist_ok=True)

    # process demonstrations
    gripper_calibrations_dirs = set()
    num_demonstrations, num_gripper_calibrations = 0, 0
    skip_demos = set()
    if cfg.skip_demos_path:
        with open(cfg.skip_demos_path, "r", encoding="ascii") as f:
            for line in f:
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                skip_demos.add(name)
        if skip_demos:
            print(f"Skipping {len(skip_demos)} demos from {cfg.skip_demos_path}")
    def process_demonstration_dir(demonstration_dir, include_gripper_calibrations=False):
        nonlocal num_demonstrations, num_gripper_calibrations
        if os.path.isdir(demonstration_dir):
            base_demo_dir_name = os.path.basename(demonstration_dir)
            session_demo_dir = os.path.join(demos_dir, base_demo_dir_name)
            if os.path.exists(session_demo_dir):
                return

            is_demonstration = base_demo_dir_name.endswith('_demonstration')
            is_gripper_calibration = base_demo_dir_name.endswith('_grippercalibration')
            
            if is_demonstration:
                if base_demo_dir_name in skip_demos:
                    print(f"Skipping abnormal demo {base_demo_dir_name}")
                    return
                if num_demonstrations >= cfg.max_demos and cfg.max_demos >= 0:
                    return

                num_demonstrations += 1

                # make sure we add the gripper calibration to the session
                for side_present in get_demonstration_sides_present(demonstration_dir):
                    json_data = get_demonstration_json_data(demonstration_dir, side_present)
                    if side_present != 'head' and 'gripperCalibrationRunName' in json_data and json_data['gripperCalibrationRunName'] != '':
                        gripper_calibrations_dirs.add(get_demonstration_path(demonstrations_dir, json_data['gripperCalibrationRunName']))
            elif is_gripper_calibration:
                if not include_gripper_calibrations:
                    return
                num_gripper_calibrations += 1
            else:
                raise NotImplementedError

            # Symlink the demonstration into the session
            symlink_absolute(demonstration_dir, session_demo_dir, target_is_directory=True)

            # print(f'Added {base_demo_dir_name} to session')

    # Copy the processed demonstrations by name filter
    for filter in cfg.input_name_filters:
        for demonstration_dir in glob(demonstrations_dir + "/*/" + filter): 
            process_demonstration_dir(demonstration_dir, include_gripper_calibrations=True)

    # Copy the processed demonstrations by session name filter
    for filter in cfg.input_session_filters:
        print(f"filter {filter}")
        for demonstration_dir in glob(demonstrations_dir + "/*/*"):
            demonstration_name = os.path.basename(demonstration_dir)
            split = demonstration_name.split('_')
            if len(split) != 4:
                continue
            demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type = split
                
            # if re.match(filter, demonstration_session_name):
            if demonstration_session_name == filter:
                print(f'Adding demonstration {demonstration_name} to session, matching session {demonstration_session_name} with filter {filter}')
                process_demonstration_dir(demonstration_dir, include_gripper_calibrations=True)

    # Copy all the associated gripper calibrations (it's possible that the gripper calibration is under a different session name or demonstration title filter that doesn't match the specified filters) so we want to manually include them
    print(gripper_calibrations_dirs)
    for demonstration_dir in gripper_calibrations_dirs:
        process_demonstration_dir(demonstration_dir, include_gripper_calibrations=True)

    print(f'Finished creating session at {session_dir} with {num_demonstrations} demonstrations and {num_gripper_calibrations} gripper calibrations')

if __name__ == '__main__':
    main()
