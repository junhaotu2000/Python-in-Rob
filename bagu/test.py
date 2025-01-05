from typing import Literal, Optional
from functools import wraps

# 定义一个装饰器，用于标记接口方法
def abstract_method(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"The method {func.__name__} must be implemented in subclasses.")
    return wrapper

class Robot:
    """Basic robot class"""
    def __init__(self, name: str, type: Literal["humanoid", "arm", "legged", "wheel"], dof: Optional[int] = None):
        self.__name = name
        self.__dof = dof

        if type not in ["humanoid", "arm", "legged", "wheel"]:
            raise ValueError(f"Invalid robot type: {type}")
        self.__type = type

    @abstract_method
    def get_state(self):
        pass

    @abstract_method
    def get_observation(self):
        pass

    @abstract_method
    def action(self):
        pass

    @abstract_method
    def reset(self):
        pass

    @abstract_method
    def shutdown(self):
        pass

# 子类 1: Arm_Robot
class Arm_Robot(Robot):
    def __init__(self, name: str, dof: int):
        super().__init__(name=name, type="arm", dof=dof)

    def get_state(self):
        return f"Arm {self._Robot__name} is in a neutral position."

    def get_observation(self):
        return {"position": [0.0] * self._Robot__dof, "force": [0.0] * self._Robot__dof}

    def action(self):
        print(f"Moving arm {self._Robot__name} to target position.")

    def reset(self):
        print(f"Resetting arm {self._Robot__name} to default position.")

    def shutdown(self):
        print(f"Shutting down arm {self._Robot__name}.")

# 子类 2: Legged_Robot
class Legged_Robot(Robot):
    def __init__(self, name: str, dof: int):
        super().__init__(name=name, type="legged", dof=dof)

    def get_state(self):
        return f"Legged {self._Robot__name} is walking."

    def get_observation(self):
        return {"joint_angles": [45.0] * self._Robot__dof, "velocity": [1.0] * self._Robot__dof}

    def action(self):
        print(f"Legged {self._Robot__name} is running to the destination.")

    def reset(self):
        print(f"Resetting legged {self._Robot__name} to standing position.")

    def shutdown(self):
        print(f"Shutting down legged {self._Robot__name}.")

# 子类 3: Humanoid (多重继承)
class Humanoid(Arm_Robot, Legged_Robot):
    def __init__(self, name: str, dof: int):
        # 使用 super() 初始化，并手动传递所需参数
        Robot.__init__(self, name=name, type="humanoid", dof=dof)

    def get_state(self):
        arm_state = Arm_Robot.get_state(self)
        leg_state = Legged_Robot.get_state(self)
        return f"Humanoid {self._Robot__name} states: [Arm: {arm_state}, Legs: {leg_state}]"

    def get_observation(self):
        arm_obs = Arm_Robot.get_observation(self)
        leg_obs = Legged_Robot.get_observation(self)
        return {
            "arm": arm_obs,
            "legs": leg_obs,
        }

    def action(self):
        Arm_Robot.action(self)
        Legged_Robot.action(self)
        print(f"Humanoid {self._Robot__name} is performing a combined action.")

    def reset(self):
        Arm_Robot.reset(self)
        Legged_Robot.reset(self)
        print(f"Humanoid {self._Robot__name} fully reset.")

    def shutdown(self):
        Arm_Robot.shutdown(self)
        Legged_Robot.shutdown(self)
        print(f"Humanoid {self._Robot__name} fully shut down.")

# 子类 4: Wheel_Robot
class Wheel_Robot(Robot):
    def __init__(self, name: str):
        super().__init__(name=name, type="wheel", dof=4)

    def get_state(self):
        return f"Wheel robot {self._Robot__name} is rolling smoothly."

    def get_observation(self):
        return {"wheel_speeds": [10.0] * self._Robot__dof}

    def action(self):
        print(f"Wheel robot {self._Robot__name} is driving to the target location.")

    def reset(self):
        print(f"Resetting wheel robot {self._Robot__name}.")

    def shutdown(self):
        print(f"Shutting down wheel robot {self._Robot__name}.")

class Robotfactory:
    @staticmethod
    def create_robot(name: str, type: Literal["humanoid", "arm", "legged", "wheel"], dof: Optional[int] = None):
        if type == "humanoid":
            return Humanoid(name, dof)
        elif type == "arm":
            return Arm_Robot(name, dof)
        elif type == "legged":
            return Legged_Robot(name, dof)
        elif type == "wheel":
            return Wheel_Robot(name)
        else:
            raise ValueError(f"Invalid robot type: {type}")

# 测试代码
if __name__ == "__main__":
    arm_robot = Arm_Robot("ArmBot", dof=7)
    legged_robot = Legged_Robot("LegBot", dof=6)
    humanoid_robot = Humanoid("HumanoidBot", dof = 30)
    wheel_robot = Wheel_Robot("WheelBot")

    for robot in [arm_robot, legged_robot, humanoid_robot, wheel_robot]:
        print(robot.get_state())
        robot.action()
        robot.reset()
        robot.shutdown()
