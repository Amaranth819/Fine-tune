import pybullet as pb
import time
import numpy as np
from pybulletgym.envs.roboschool.envs.locomotion.humanoid_env import HumanoidBulletEnv
from scipy.stats import vonmises_line
from math import floor


'''
    Reward function:
    (1) Ground reaction force:
        E[C_frc(phi)] = c_swing_frc * E[I_swing_frc(phi)] + c_stance_frc * E[I_stance_frc(phi)]
    (2) Foot velocity:
        E[C_spd(phi)] = c_swing_spd * E[I_swing_spd(phi)] + c_stance_spd * E[I_stance_spd(phi)]
    
    E[R(s,phi)] = E[C_frc(phi + shift_left)] * q_left_frc(s)
                + E[C_frc(phi + shift_right)] * q_right_frc(s)
                + E[C_spd(phi + shift_left)] * q_left_spd(s)
                + E[C_spd(phi + shift_right)] * q_right_spd(s)
'''


class PeriodicHumanoidPyBulletEnv(HumanoidBulletEnv):
    def __init__(self, period_length = 1, gait_param_dict = {'stance_ratio' : 0.5, 'kappa' : 8}):
        super().__init__()

        # Gait parameter dictionary
        self.P_I_stance, self.P_I_swing = self._PeriodicGaitInit(gait_param_dict)
        
        # Expectation
        self.E_C_frc_stance = lambda phi: -1 * (1 * self.P_I_stance(phi, 1))
        self.E_C_frc_swing = lambda phi: -1 * (1 * self.P_I_swing(phi, 1))
        self.E_C_spd_stance = lambda phi: -1 * (1 * self.P_I_stance(phi, 0))
        self.E_C_spd_swing = lambda phi: -1 * (1 * self.P_I_swing(phi, 0))

        # Period
        self.period_length = period_length
        # self.simultion_timestep = 
        self.current_time = 0.0


    '''
        Time and phase information
    '''
    def StepSimulationTime(self):
        self.current_time += self.scene.timestep * self.scene.frame_skip


    def GetCurrentSimulationTime(self):
        return self.current_time


    def GetCurrentPhase(self):
        # Output: phase in [0, 1]
        return (self.current_time % self.period_length) / self.period_length


    '''
        Dynamic information
    '''
    def GetFootVelocity(self):
        foot_velocity = {}

        # foot_list: "right_foot", "left_foot"
        for foot_name in self.robot.foot_list:
            linvel, _ = self.robot.parts[foot_name].get_velocity() # In Cartesian world coordinate
            foot_velocity[foot_name] = np.linalg.norm(linvel)

        return foot_velocity


    def GetFootGroundReactionForces(self):
        foot_GRF = {}

        for foot_name in self.robot.foot_list:
            contact_list = self.robot.parts[foot_name].contact_list()
            foot_GRF[foot_name] = self._GetTotalGRFFromContactList(contact_list)

        return foot_GRF


    def _GetTotalGRFFromContactList(self, contact_list):
        total_grf = np.zeros(3, dtype = np.float32)

        for contact in contact_list:
            contactNormalOnGround, normalForce = contact[7], contact[9]
            total_grf += np.array(contactNormalOnGround) * normalForce

        # The direction of GRF is always perpendicular to the flat ground
        grf_magnitude = np.linalg.norm(total_grf)

        return grf_magnitude

    
    '''
        Periodic gait setting
    '''
    def _PeriodicGaitInit(self, gait_param_dict):
        # [0, 1]
        stance_ratio = gait_param_dict['stance_ratio']
        assert stance_ratio > 0 and stance_ratio < 1
        kappa = gait_param_dict['kappa']

        # Stance parameters
        stance_duration = stance_ratio
        stance_start_mean = 0
        stance_end_mean = stance_start_mean + stance_duration
        next_stance_start_mean = 1
        next_stance_end_mean = next_stance_start_mean + stance_duration

        # Swing parameters
        swing_duration = 1 - stance_duration
        swing_start_mean = stance_end_mean
        swing_end_mean = swing_start_mean + swing_duration
        last_swing_end_mean = 0
        last_swing_start_mean = last_swing_end_mean - swing_duration


        def P_I_stance(phi, I_stance):
            phi = self._CycleTimeToVonMiseInput(phi)

            P1 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(stance_start_mean))
            P2 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(stance_end_mean))

            P3 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(next_stance_start_mean))
            P4 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(next_stance_end_mean))    

            if I_stance == 1:
                return P1 * (1 - P2) + P3 * (1 - P4)
            else:
                return 1 - (P1 * (1 - P2) + P3 * (1 - P4))


        def P_I_swing(phi, I_swing):
            phi = self._CycleTimeToVonMiseInput(phi)

            P1 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(swing_start_mean))
            P2 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(swing_end_mean))

            P3 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(last_swing_start_mean))
            P4 = vonmises_line.cdf(phi, kappa, self._CycleTimeToVonMiseInput(last_swing_end_mean))
            
            if I_swing == 1:
                return P1 * (1 - P2) + P3 * (1 - P4)
            else:
                return 1 - (P1 * (1 - P2) + P3 * (1 - P4))
        

        # def expectation_stance(phi):
        #     return -1 * (1 * P_I_stance(phi, 1) + 0 * P_I_stance(phi, 0))


        # def expectation_swing(phi):
        #     return -1 * (1 * P_I_swing(phi, 1) + 0 * P_I_swing(phi, 0))


        return P_I_stance, P_I_swing


    def _CycleTimeToVonMiseInput(self, phi):
        # Input: phi in [0, 1]
        # Output: [-pi, pi]
        return -np.pi + 2 * np.pi * phi


    '''
        Step the simulation
    '''
    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

            # Update the simulation time
            self.StepSimulationTime()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit        

        '''
            Reward
        '''
        # Alive bonus
        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        # Phase: Assume the left leg swings first
        phi = self.GetCurrentPhase()
        GRFs = self.GetFootGroundReactionForces()
        FootVels = self.GetFootVelocity()

        # print(GRFs, FootVels)
        lf_frc_reward = 0.1 * self.E_C_frc_swing(phi) * GRFs['left_foot']
        rf_frc_reward = 0.1 * self.E_C_frc_stance(phi) * GRFs['right_foot']

        lf_spd_reward = 1 * self.E_C_spd_swing(phi) * FootVels['left_foot']
        rf_spd_reward = 1 * self.E_C_spd_stance(phi) * FootVels['right_foot']

        bipedal_reward = lf_frc_reward + rf_frc_reward + lf_spd_reward + rf_spd_reward

        # Energy cost
        electricity_cost = self.electricity_cost * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        # Forward velocity
        forward_speed = state[3:6]
        forward_speed_reward = 1.0 * np.linalg.norm(forward_speed)



        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("lf_frc_reward")
            print(lf_frc_reward)
            print("rf_frc_reward")
            print(rf_frc_reward)
            print("lf_spd_reward")
            print(lf_spd_reward)
            print("rf_spd_reward")
            print(rf_spd_reward)
            print("electricity_cost")
            print(electricity_cost)
            print("forward_speed_reward")
            print(forward_speed_reward)

        self.rewards = [
            alive,
            lf_frc_reward,
            rf_frc_reward,
            lf_spd_reward,
            rf_spd_reward,
            electricity_cost,
            forward_speed_reward
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        # print(self.rewards)


        # print(self.rewards)

        return state, sum(self.rewards), bool(done), {}

        
        



# if __name__ == '__main__':
#     env = PeriodicHumanoidPyBulletEnv()
#     env.render('human')
#     env.reset()

#     for _ in range(300):
#         state, r, _, _ = env.step(env.action_space.sample())
#         env.render('human')

#         # for i, foot in enumerate(env.robot.feet):
#         #     # print(i, foot.bodyIndex)
#         #     print(i, foot.contact_list()) 

#         time.sleep(0.1)

    # print(env.robot.feet)
    # print(env.robot.foot_list)
    # print(env.robot.parts.keys())

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(1, 1)
    # phi_list = np.linspace(0, 1, 100)
    # # ax.plot(phi_list, [env.E_C_frc_swing(phi) for phi in phi_list], label = 'E_C_frc_swing')
    # ax.plot(phi_list, [env.E_C_spd_swing(phi) for phi in phi_list], label = 'E_C_spd_swing')
    # # ax.plot(phi_list, [env.E_C_frc_stance(phi) for phi in phi_list], label = 'E_C_frc_stance')
    # ax.plot(phi_list, [env.E_C_spd_stance(phi) for phi in phi_list], label = 'E_C_spd_stance')

    # plt.legend()
    # plt.show()