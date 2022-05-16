from distutils.log import info
import pybullet as pb
import time
import numpy as np
from pybulletgym.envs.roboschool.envs.locomotion.humanoid_env import HumanoidBulletEnv
from scipy.stats import vonmises_line, vonmises
from math import floor, pi, exp


class PeriodicHumanoidPyBulletEnv2(HumanoidBulletEnv):
    def __init__(self, 
        period_length = 1, 
        vonmises_params = {'period' : 1, 'ratio' : 0.5, 'kappa' : 64}
    ):
        super().__init__()

        # Parameters
        self.period_length = period_length
        self.current_time = 0.0
        
        self.period = vonmises_params['period']
        self.ratio = vonmises_params['ratio']
        self.kappa = vonmises_params['kappa']

        # Reward coefficients
        self.C_swing_frc = 1
        self.C_stance_frc = 1
        self.C_swing_spd = 1
        self.C_stance_spd = 1

        # Rewward functions
        # Swing order: LF -> RF -> LF -> RF -> ...
        self.E_C_LF_frc = lambda t: self.E_C(t, shift = 0.0)
        self.E_C_RF_frc = lambda t: self.E_C(t, shift = 0.5)
        self.E_C_LF_spd = lambda t: self.E_C(t, shift = 0.5)
        self.E_C_RF_spd = lambda t: self.E_C(t, shift = 0.0)
        
        print('PeriodicHumanoidPyBulletEnv2')

    
    '''
        Time and phase information
    '''
    def StepSimulationTime(self):
        self.current_time += self.scene.timestep * self.scene.frame_skip


    def GetCurrentSimulationTime(self):
        return self.current_time


    # def GetCurrentPhase(self):
    #     # Output: phase in [0, 1]
    #     return (self.current_time % self.period_length) / self.period_length


    '''
        Von Mises expectation
    '''
    def _PeriodicVonMiseDist(self, t, I, shift = 0):
        phi_t = (t % self.period) / self.period + shift

        P1 = vonmises_line.cdf(2*pi*phi_t, self.kappa, 0)
        P2 = vonmises_line.cdf(2*pi*phi_t, self.kappa, 2*pi*self.ratio)
        P3 = vonmises_line.cdf(2*pi*phi_t, self.kappa, 2*pi)
        P4 = vonmises_line.cdf(2*pi*phi_t, self.kappa, 2*pi*(1 + self.ratio))

        if I == 1:
            return P1 * (1 - P2) + P3 * (1 - P4)
        else:
            return 1 - (P1 * (1 - P2) + P3 * (1 - P4))


    def E_C(self, t, shift = 0):
        # Shift: [0, 1]
        return -1 * 1 * self._PeriodicVonMiseDist(t, 1, shift)


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
        Step the simulation
    '''
    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -100.0  # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying


    def step(self, a):
        # Decode the action variable

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
        alive = float(self.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        # Phase: Assume the left leg swings first
        t = self.current_time
        GRFs = self.GetFootGroundReactionForces()
        FootVels = self.GetFootVelocity()

        # print(GRFs, FootVels)
        omega = 0.5

        def q_frc(frc):
            return 1 - exp(-omega * frc**2 / 100)

        def q_spd(spd):
            return 1 - exp(-2 * omega * spd**2)


        lf_frc_reward = self.E_C_LF_frc(t) * q_frc(GRFs['left_foot'])
        rf_frc_reward = self.E_C_RF_frc(t) * q_frc(GRFs['right_foot'])
        lf_spd_reward = self.E_C_LF_spd(t) * q_spd(FootVels['left_foot'])
        rf_spd_reward = self.E_C_RF_spd(t) * q_spd(FootVels['right_foot'])

        bipedal_reward = 0.4 * (lf_frc_reward + rf_frc_reward + lf_spd_reward + rf_spd_reward)

        # Energy cost
        electricity_cost = self.electricity_cost * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        # Forward velocity
        # forward_speed = state[3:6]
        # forward_speed_reward = -1.0 * (np.linalg.norm(forward_speed) - 1.0)**2 # Set the desired velocity to 1 first. Punish the difference.
        pelvis_speed, _ = self.robot.parts['pelvis'].get_velocity()
        pelvis_speed_reward = 0.3 * -(np.linalg.norm(pelvis_speed) - 1.0)**2


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
            print("pelvis_speed_reward")
            print(pelvis_speed_reward)

        self.rewards = [
            alive,
            bipedal_reward,
            electricity_cost,
            pelvis_speed_reward
        ]

        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)


        # info = {}
        info = {
            'alive' : alive,
            # 'lf_frc_reward' : lf_frc_reward,
            # 'rf_frc_reward' : rf_frc_reward,
            # 'lf_spd_reward' : lf_spd_reward,
            # 'rf_spd_reward' : rf_spd_reward,
            'bipedal_reward' : bipedal_reward,
            'electricity_cost' : electricity_cost,
            'pelvis_speed_reward' : pelvis_speed_reward,
            'total_reward' : sum(self.rewards)
        }


        return state, sum(self.rewards), bool(done), info



# if __name__ == '__main__':
    
#     env = PeriodicHumanoidPyBulletEnv2(
#         vonmises_params = {'period' : 1, 'ratio' : 0.5, 'kappa' : 64})
#     import matplotlib.pyplot as plt

#     t_list = np.linspace(0, 4, 1000)

#     fig, ax = plt.subplots(2)
#     fig.set_size_inches(20, 6)

#     ax[0].plot(t_list, [env.E_C_LF_frc(t) for t in t_list], label = 'E_C_LF_frc')
#     ax[0].plot(t_list, [env.E_C_RF_frc(t) for t in t_list], label = 'E_C_RF_frc')
#     ax[0].legend(loc = 1)
    
#     ax[1].plot(t_list, [env.E_C_LF_spd(t) for t in t_list], label = 'E_C_LF_spd')
#     ax[1].plot(t_list, [env.E_C_RF_spd(t) for t in t_list], label = 'E_C_RF_spd')
#     ax[1].legend(loc = 1)
#     plt.show()