# Xbox Controller Integration Examples

This document shows how to use the **clean configuration-based** Xbox controller integration with LeRobot.

## **Clean Architecture**

✅ **Single Source of Truth**:
- `controllers/configs.py` - **Main Xbox configuration** (XboxControllerConfig)
- `controllers/xbox_controller.py` - Xbox controller logic
- `robots/manipulator.py` - Enhanced with Xbox support
- `scripts/control_robot.py` - CLI integration

❌ **No More Duplication**:
- Xbox parameters only defined once in `controllers/configs.py`
- Control configs use composition: `xbox_controller: XboxControllerConfig | None`

## **Simple Xbox Teleoperation**

```bash
# Standard teleoperation with Xbox controller enabled
python lerobot/scripts/control_robot.py \
    --robot.type so101 \
    --control.type teleoperate \
    --control.xbox_controller.device_index 0 \
    --control.xbox_target_arm main \
    --robot.cameras '{}'

# With custom Xbox settings
python lerobot/scripts/control_robot.py \
    --robot.type so101 \
    --control.type teleoperate \
    --control.xbox_controller.device_index 0 \
    --control.xbox_controller.gripper_speed 0.05 \
    --control.xbox_controller.dead_zone_sticks 0.15 \
    --control.xbox_target_arm main \
    --robot.leader_arms '{}' \
    --robot.cameras '{}'
```

## **Xbox Recording**

```bash
# Record dataset using Xbox controller
python lerobot/scripts/control_robot.py \
    --robot.type so101 \
    --control.type record \
    --control.repo_id your_username/xbox_dataset \
    --control.single_task "Pick and place with Xbox controller" \
    --control.num_episodes 10 \
    --control.xbox_controller.device_index 0 \
    --robot.cameras '{}'

# Advanced Xbox recording with custom settings
python lerobot/scripts/control_robot.py \
    --robot.type aloha \
    --control.type record \
    --control.repo_id your_username/xbox_aloha \
    --control.single_task "Xbox controlled manipulation" \
    --control.xbox_controller.device_index 0 \
    --control.xbox_controller.x_axis_speed 0.03 \
    --control.xbox_controller.y_axis_speed 0.03 \
    --control.xbox_target_arm left \
    --control.num_episodes 50
```

## **Mixed Control (Xbox + Leader Arms)**

For dual-arm robots like ALOHA:

```bash
# Xbox controls left arm, leader arm controls right arm
python lerobot/scripts/control_robot.py \
    --robot.type aloha \
    --control.type teleoperate \
    --control.xbox_controller.device_index 0 \
    --control.xbox_target_arm left
    # Right arm automatically uses its leader arm

# Xbox controls right arm, leader arm controls left arm  
python lerobot/scripts/control_robot.py \
    --robot.type aloha \
    --control.type teleoperate \
    --control.xbox_controller.device_index 0 \
    --control.xbox_target_arm right
    # Left arm automatically uses its leader arm
```

## **Default Behavior**

```bash
# If xbox_controller is NOT specified, it defaults to leader arms
python lerobot/scripts/control_robot.py \
    --robot.type aloha \
    --control.type teleoperate
    # Both arms use leader arms (default behavior)

# These are equivalent (no Xbox controller):
python lerobot/scripts/control_robot.py \
    --robot.type aloha \
    --control.type teleoperate
    # xbox_controller defaults to None
```

## **Available Xbox Parameters**

All Xbox controller settings are defined once in `controllers/configs.py` and accessed via:

```bash
--control.xbox_controller.device_index 0               # Controller index (0, 1, 2...)
--control.xbox_controller.dead_zone_sticks 0.1          # Stick deadzone
--control.xbox_controller.dead_zone_triggers 0.1        # Trigger deadzone
--control.xbox_controller.shoulder_pan_speed 0.025      # Movement speeds
--control.xbox_controller.x_axis_speed 0.025
--control.xbox_controller.y_axis_speed 0.025
--control.xbox_controller.wrist_flex_speed 0.025
--control.xbox_controller.wrist_roll_speed 0.025
--control.xbox_controller.gripper_speed 0.03

# Plus target arm selection:
--control.xbox_target_arm main                          # Target arm ("main", "left", "right")
```

## **Key Benefits of Configuration-Based Architecture**

1. **✅ Single Source of Truth**: Xbox config only in `controllers/configs.py`
2. **✅ No Duplication**: Same config used everywhere
3. **✅ Type Safety**: Full dataclass with validation
4. **✅ Discoverable**: All options visible in `--help`
5. **✅ Flexible**: Mix and match any Xbox settings
6. **✅ Clean Composition**: `xbox_controller: XboxControllerConfig | None`
7. **✅ Backward Compatible**: Default behavior unchanged (`None` = no Xbox)

## **Control Mode Summary**

| Scenario | xbox_controller | xbox_target_arm | Result |
|----------|----------------|------------------|---------|
| **Default** | `None` (omitted) | - | All arms use leader arms |
| **Single Xbox** | `{device_index: 0}` | `left` | Left: Xbox, Right: Leader |
| **Single Xbox** | `{device_index: 0}` | `right` | Right: Xbox, Left: Leader |
| **Single Xbox** | `{device_index: 0}` | `main` | Main arm: Xbox |

## **Implementation Details**

### **How It Works**
1. **Single Config Class**: XboxControllerConfig in `controllers/configs.py`
2. **Composition**: Control configs have `xbox_controller: XboxControllerConfig | None`
3. **Enhanced ManipulatorRobot**: Xbox support built into base class
4. **Seamless Fallback**: Falls back to leader arms if Xbox is None

### **File Structure**
```
lerobot/common/robot_devices/
├── controllers/
│   ├── xbox_controller.py    # Xbox controller logic
│   └── configs.py           # ⭐ SINGLE SOURCE OF TRUTH ⭐
└── robots/
    └── manipulator.py       # Enhanced with Xbox support
```

### **Configuration Flow**
1. **CLI**: `--control.xbox_controller.device_index 0`
2. **Parser**: Creates XboxControllerConfig object
3. **Control**: TeleoperateControlConfig.xbox_controller = XboxControllerConfig(...)
4. **Robot**: ManipulatorRobot uses the config directly

This **configuration-based architecture** eliminates duplication while providing clean, type-safe Xbox functionality that's fully integrated with LeRobot's design patterns. 