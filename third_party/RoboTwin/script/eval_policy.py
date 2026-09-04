import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
import json

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    module_path = Path(envs_module.__file__).resolve()
    expected_root = Path(
        os.environ.get("FASTWAM_ROBOTWIN_SOURCE_OF_TRUTH", Path.cwd())
    ).resolve()
    if not module_path.is_relative_to(expected_root):
        raise RuntimeError(
            f"RoboTwin task import escaped source of truth: {module_path} "
            f"is not under {expected_root}"
        )
    print(f"ROBOTWIN_TASK_IMPORT={module_path}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    if usr_args.get("minibench_condition") is not None:
        args["minibench_condition"] = usr_args["minibench_condition"]

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan,
                                   seed_manifest_path=usr_args.get("seed_manifest_path"),
                                   episode_output_dir=save_dir)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                seed_manifest_path=None,
                episode_output_dir=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    suc_test_seed_list = []
    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    strict_seeds = None
    if seed_manifest_path is not None and str(seed_manifest_path).strip():
        manifest_path = Path(str(seed_manifest_path)).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_task = manifest.get("task")
        if manifest_task is not None and manifest_task != task_name:
            raise ValueError(
                f"Seed manifest task={manifest_task!r} does not match evaluation task={task_name!r}"
            )
        manifest_config = manifest.get("task_config")
        if manifest_config is not None and manifest_config != args["task_config"]:
            raise ValueError(
                f"Seed manifest task_config={manifest_config!r} does not match "
                f"evaluation config={args['task_config']!r}"
            )
        strict_seeds = [int(seed) for seed in manifest["seeds"]]
        if len(strict_seeds) != test_num:
            raise ValueError(
                f"Seed manifest has {len(strict_seeds)} seeds but eval_num_episodes={test_num}: "
                f"{manifest_path}"
            )
        if len(strict_seeds) != len(set(strict_seeds)):
            raise ValueError(f"Seed manifest contains duplicate seeds: {manifest_path}")
        print(f"STRICT_SEED_MANIFEST={manifest_path}")
        print(f"STRICT_SEEDS={strict_seeds}")
    if task_name == "move_block_reveal_can" and strict_seeds is None:
        raise ValueError(
            "move_block_reveal_can requires seed_manifest_path; implicit seed "
            "substitution is forbidden for the paired experiment"
        )

    def close_task(clear_cache=False):
        try:
            TASK_ENV.close_env(clear_cache=clear_cache)
        except Exception:
            pass

    def prepare_handoff():
        hook = getattr(TASK_ENV, "prepare_policy_handoff", None)
        if hook is not None:
            hook()

    if episode_output_dir is None:
        raise ValueError("episode_output_dir is required for deterministic result logging")
    episode_output_dir = Path(episode_output_dir)

    def append_episode_record(payload):
        path = episode_output_dir / "episodes.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    prevalidated_strict_tasks = {
        "place_can_basket",
        "move_block_reveal_can_block_calibration",
        "move_block_reveal_can",
    }

    def episode_info_from_frozen_scene():
        if task_name in {
            "move_block_reveal_can",
            "move_block_reveal_can_block_calibration",
        }:
            return {"info": {}}
        if task_name == "place_can_basket":
            return {
                "info": {
                    "{A}": f"{TASK_ENV.can_name}/base{TASK_ENV.can_id}",
                    "{B}": f"{TASK_ENV.basket_name}/base{TASK_ENV.basket_id}",
                    "{a}": str(TASK_ENV.arm_tag),
                }
            }
        raise RuntimeError(
            "No deterministic instruction metadata adapter for prevalidated "
            f"strict task {task_name!r}"
        )

    for now_id in range(test_num):
        requested_seed = strict_seeds[now_id] if strict_seeds is not None else None
        expert_attempt = 0
        expert_attempts_used = 0
        expert_precheck_skipped = bool(
            task_name in prevalidated_strict_tasks and requested_seed is not None
        )
        max_expert_attempts = int(
            getattr(TASK_ENV, "SCRIPTED_EXPERT_MAX_ATTEMPTS", 1)
        )
        if max_expert_attempts < 1:
            raise ValueError("SCRIPTED_EXPERT_MAX_ATTEMPTS must be positive")

        if expert_precheck_skipped:
            # The frozen MiniBench protocol validates these exact paired
            # task and calibration scenes with a separate simulator-only
            # expert audit. Re-running the stochastic CuRobo expert here would
            # re-screen fixed seeds at policy-evaluation time and could abort
            # before the model sees the scene. Evaluate every manifest seed
            # directly instead.
            now_seed = requested_seed
            episode_info = None
            print(
                "SCRIPTED_EXPERT_PREFLIGHT_SKIPPED "
                f"task={task_name} seed={now_seed} "
                "reason=frozen_simulator_audit"
            )
        else:
            # Native RoboTwin evaluation normally skips seeds its expert cannot
            # solve.  A strict manifest instead fails closed so calibration
            # seeds can never silently drift to different seeds.
            while True:
                now_seed = requested_seed if requested_seed is not None else now_seed
                expert_error = None
                expert_traceback = None
                render_freq = args["render_freq"]
                args["render_freq"] = 0
                try:
                    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                    prepare_handoff()
                    episode_info = TASK_ENV.play_once()
                    expert_ok = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
                except Exception as exc:
                    expert_ok = False
                    expert_error = f"{type(exc).__name__}: {exc}"
                    expert_traceback = traceback.format_exc()
                finally:
                    close_task()
                    args["render_freq"] = render_freq

                expert_attempts_used = expert_attempt + 1
                if expert_ok:
                    break
                if requested_seed is not None:
                    expert_attempt += 1
                    if expert_attempt < max_expert_attempts:
                        print(
                            "STRICT_SEED_EXPERT_RETRY "
                            f"task={task_name} seed={now_seed} "
                            f"next_attempt={expert_attempt + 1}/{max_expert_attempts} "
                            f"reason={expert_error or 'plan_success/check_success false'}"
                        )
                        continue
                    raise RuntimeError(
                        f"Strict seed {now_seed} failed scripted expert for {task_name}: "
                        f"{expert_error or 'plan_success/check_success false'} "
                        f"after {max_expert_attempts} fixed-seed attempts\n"
                        f"{expert_traceback or ''}"
                    )
                now_seed += 1

        suc_test_seed_list.append(now_seed)
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        prepare_handoff()

        if episode_info is None:
            episode_info = episode_info_from_frozen_scene()

        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        if not results or not results[0][instruction_type]:
            close_task()
            raise RuntimeError(
                f"No {instruction_type!r} instruction generated for task={task_name}, seed={now_seed}"
            )
        instruction = str(np.random.choice(results[0][instruction_type]))
        TASK_ENV.set_instruction(instruction=instruction)
        print(f"EPISODE_START index={now_id} seed={now_seed} instruction={instruction!r}")

        current_video_path = None
        renamed_video_path = None
        ffmpeg_process = None
        if TASK_ENV.eval_video_path is not None:
            episode_idx = TASK_ENV.test_num
            current_video_path = Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", video_size,
                    "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264", "-crf", "23", str(current_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            ffmpeg_process = ffmpeg
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        video_finalized = False
        try:
            reset_func(model)
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                need_obs = True
                if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                    need_obs = bool(model.should_request_observation())
                observation = TASK_ENV.get_obs() if need_obs else None
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break
            succ = bool(succ or TASK_ENV.check_success())

            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()
                video_finalized = True
                if ffmpeg_process is None or ffmpeg_process.returncode != 0:
                    raise RuntimeError(
                        "ffmpeg failed while finalizing the episode video: "
                        f"returncode={None if ffmpeg_process is None else ffmpeg_process.returncode}"
                    )
                if current_video_path is None or not current_video_path.exists():
                    raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
                if current_video_path.stat().st_size <= 0:
                    raise RuntimeError(f"Episode video is empty: {current_video_path}")
                is_randomized = "randomized" in str(args["task_config"]).lower()
                renamed_video_path = (
                    Path(TASK_ENV.eval_video_path)
                    / f"episode{episode_idx}_seed-{now_seed}_randomized-"
                    f"{str(is_randomized).lower()}_success-{str(succ).lower()}.mp4"
                )
                current_video_path.rename(renamed_video_path)

            if hasattr(TASK_ENV, "get_minibench_record"):
                episode_record = TASK_ENV.get_minibench_record(refresh_visibility=True)
            else:
                episode_record = {
                    "task": task_name,
                    "condition": None,
                    "seed": now_seed,
                }
            episode_record.update(
                {
                    "episode_index": now_id,
                    "seed": now_seed,
                    "scripted_expert_precheck_skipped": expert_precheck_skipped,
                    "scripted_expert_attempts_used": expert_attempts_used,
                    "scripted_expert_max_attempts": max_expert_attempts,
                    "instruction": instruction,
                    "success": succ,
                    "video_path": str(renamed_video_path) if renamed_video_path else None,
                }
            )
            append_episode_record(episode_record)
        finally:
            if not video_finalized:
                ffmpeg_process = getattr(TASK_ENV, "eval_video_ffmpeg", None)
                if ffmpeg_process is not None:
                    try:
                        TASK_ENV._del_eval_video_ffmpeg()
                    except Exception:
                        try:
                            if ffmpeg_process.stdin is not None and not ffmpeg_process.stdin.closed:
                                ffmpeg_process.stdin.close()
                            ffmpeg_process.terminate()
                            ffmpeg_process.wait(timeout=5)
                        except Exception:
                            ffmpeg_process.kill()
                            ffmpeg_process.wait()
            close_task(clear_cache=((now_id + 1) % clear_cache_freq == 0))

        TASK_ENV.test_num += 1
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | seed={now_seed}\n"
            f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} "
            f"=> {round(TASK_ENV.suc / TASK_ENV.test_num * 100, 1)}%\n"
        )
        if requested_seed is None:
            now_seed += 1

    print(f"EVALUATED_SEEDS={suc_test_seed_list}")
    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
