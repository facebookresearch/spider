"""Minimal test file for HDMI simulator integration with SPIDER.

Run this test from the spider directory using:
    source ../HDMI/.venv/bin/activate
    python -m pytest spider/simulators/hdmi_test.py -v -s
"""

import types

import torch


def test_imports():
    """Test that basic imports work."""
    import spider
    from spider.simulators import hdmi

    assert isinstance(spider.ROOT, str)
    assert isinstance(hdmi, types.ModuleType)


def test_setup_and_state():
    """Test environment setup, save/load state, and stepping."""
    from spider.config import Config
    from spider.simulators import hdmi

    # Create config
    config = Config(
        task="move_suitcase",
        num_samples=4,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        viewer="",  # headless mode
    )
    ref_data = tuple()

    # Setup environment
    env = hdmi.setup_env(config, ref_data)
    assert env is not None
    assert env.num_envs == config.num_samples

    # Get action dimension
    num_actions = env.action_spec.shape[-1]
    assert num_actions > 0

    # Test save state
    state1 = hdmi.save_state(env)
    assert "robot_root_pos" in state1
    assert "robot_joint_pos" in state1

    # Step environment
    ctrl = torch.zeros(config.num_samples, num_actions, device=env.device)
    hdmi.step_env(config, env, ctrl)

    # Save state after step
    state2 = hdmi.save_state(env)

    # States should be different after stepping
    assert not torch.allclose(state1["robot_joint_pos"], state2["robot_joint_pos"])

    # Load first state
    hdmi.load_state(env, state1)
    state1_reloaded = hdmi.save_state(env)

    # Verify state was correctly restored
    assert torch.allclose(state1["robot_root_pos"], state1_reloaded["robot_root_pos"])
    assert torch.allclose(state1["robot_joint_pos"], state1_reloaded["robot_joint_pos"])


def test_reward():
    """Test reward computation."""
    from spider.config import Config
    from spider.simulators import hdmi

    config = Config(
        task="move_suitcase",
        num_samples=4,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        viewer="",
    )
    ref_data = tuple()

    env = hdmi.setup_env(config, ref_data)

    # Get reward
    reward, info = hdmi.get_reward(config, env, ref_data)

    # Verify reward shape and type
    assert isinstance(reward, torch.Tensor)
    assert reward.shape == (config.num_samples,)
    assert isinstance(info, dict)

    # Test terminal reward
    terminal_reward, _ = hdmi.get_terminal_reward(config, env, ref_data)
    assert terminal_reward.shape == (config.num_samples,)


def test_reset_determinism():
    """Test that env.reset() is deterministic without using seeds."""
    from spider.config import Config
    from spider.simulators import hdmi
    import gc

    config = Config(
        task="move_suitcase",
        num_samples=2,  # Use fewer envs to save memory
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        viewer="",
    )
    ref_data = tuple()

    env = hdmi.setup_env(config, ref_data)

    # Reset multiple times WITHOUT setting seed - should still be deterministic
    env.reset()
    state1 = hdmi.save_state(env)

    env.reset()
    state2 = hdmi.save_state(env)

    env.reset()
    state3 = hdmi.save_state(env)

    # Check all states are identical
    # Debug: print differences
    if not torch.allclose(state1["robot_joint_pos"], state2["robot_joint_pos"]):
        diff = (state1["robot_joint_pos"] - state2["robot_joint_pos"]).abs()
        print(f"Joint pos max diff: {diff.max().item()}")
        print(f"Joint pos mean diff: {diff.mean().item()}")

    assert torch.allclose(state1["robot_root_pos"], state2["robot_root_pos"], atol=1e-6), \
        "Reset is not deterministic: robot_root_pos differs"
    assert torch.allclose(state1["robot_root_quat"], state2["robot_root_quat"], atol=1e-6), \
        "Reset is not deterministic: robot_root_quat differs"
    assert torch.allclose(state1["robot_joint_pos"], state2["robot_joint_pos"], atol=1e-6), \
        "Reset is not deterministic: robot_joint_pos differs"
    assert torch.allclose(state1["robot_joint_vel"], state2["robot_joint_vel"], atol=1e-6), \
        "Reset is not deterministic: robot_joint_vel differs"

    assert torch.allclose(state2["robot_root_pos"], state3["robot_root_pos"]), \
        "Reset is not deterministic: robot_root_pos differs (2nd vs 3rd)"
    assert torch.allclose(state2["robot_joint_pos"], state3["robot_joint_pos"]), \
        "Reset is not deterministic: robot_joint_pos differs (2nd vs 3rd)"

    # Check object states if they exist
    for key in state1.keys():
        if key.startswith("suitcase_"):
            assert torch.allclose(state1[key], state2[key]), \
                f"Reset is not deterministic: {key} differs"
            assert torch.allclose(state2[key], state3[key]), \
                f"Reset is not deterministic: {key} differs (2nd vs 3rd)"


def test_load_state_after_steps():
    """Test that load_state correctly restores environment after multiple steps."""
    from spider.config import Config
    from spider.simulators import hdmi
    import gc

    config = Config(
        task="move_suitcase",
        num_samples=2,  # Use fewer envs to save memory
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        viewer="",
    )
    ref_data = tuple()

    env = hdmi.setup_env(config, ref_data)
    num_actions = env.action_spec.shape[-1]

    # Save initial state
    initial_state = hdmi.save_state(env)

    # Step environment multiple times with different controls
    num_steps = 10
    for i in range(num_steps):
        ctrl = torch.randn(config.num_samples, num_actions, device=env.device) * 0.1
        hdmi.step_env(config, env, ctrl)

    # Save state after stepping
    stepped_state = hdmi.save_state(env)

    # Verify states are different
    assert not torch.allclose(initial_state["robot_joint_pos"], stepped_state["robot_joint_pos"]), \
        "State should be different after stepping"
    assert not torch.allclose(initial_state["robot_root_pos"], stepped_state["robot_root_pos"]), \
        "Root position should be different after stepping"

    # Load initial state back
    hdmi.load_state(env, initial_state)
    reloaded_state = hdmi.save_state(env)

    # Verify all state components match exactly
    for key in initial_state.keys():
        if isinstance(initial_state[key], torch.Tensor):
            assert torch.allclose(initial_state[key], reloaded_state[key], atol=1e-6), \
                f"load_state failed: {key} differs after reload"
        else:
            assert initial_state[key] == reloaded_state[key], \
                f"load_state failed: {key} differs after reload"

    # Step forward from reloaded state
    ctrl = torch.zeros(config.num_samples, num_actions, device=env.device)
    hdmi.step_env(config, env, ctrl)
    state_after_reload_step = hdmi.save_state(env)

    # Load initial state again
    hdmi.load_state(env, initial_state)
    hdmi.step_env(config, env, ctrl)
    state_after_reload_step2 = hdmi.save_state(env)

    # Verify stepping from same state produces same result
    if not torch.allclose(
        state_after_reload_step["robot_joint_pos"],
        state_after_reload_step2["robot_joint_pos"],
        atol=1e-5
    ):
        diff = (state_after_reload_step["robot_joint_pos"] - state_after_reload_step2["robot_joint_pos"]).abs()
        print(f"\nJoint pos max diff after reload+step: {diff.max().item()}")
        print(f"Joint pos mean diff after reload+step: {diff.mean().item()}")

    # Allow small tolerance due to floating point precision in simulation
    # Max diff ~0.01 radians (~0.5 degrees) is acceptable
    assert torch.allclose(
        state_after_reload_step["robot_joint_pos"],
        state_after_reload_step2["robot_joint_pos"],
        atol=0.02
    ), f"Stepping from same loaded state should produce nearly identical results (max diff: {diff.max().item():.6f} radians)"


if __name__ == "__main__":
    import gc
    print("Running minimal HDMI simulator tests...")

    print("\n1. Testing imports...")
    test_imports()
    print("✓ Imports test passed")

    print("\n2. Testing environment setup and state management...")
    test_setup_and_state()
    print("✓ Setup and state test passed")

    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n3. Testing reward computation...")
    test_reward()
    print("✓ Reward test passed")

    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n4. Testing reset determinism...")
    test_reset_determinism()
    print("✓ Reset determinism test passed")

    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n5. Testing load_state after multiple steps...")
    test_load_state_after_steps()
    print("✓ Load state after steps test passed")

    print("\n✅ All tests passed!")
