import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import vonmises, norm
from scipy.stats import vonmises_line
from scipy.special import i0



# A_kappa = 8
# A_loc = 0.3
# A = np.linspace(vonmises.ppf(0.01, A_kappa, loc = A_loc), vonmises.ppf(0.99, A_kappa, loc = A_loc), 100)



# B_kappa = 1
# B_loc = 0.7 * np.pi
# B = np.linspace(vonmises.ppf(0.01, B_kappa, loc = B_loc), vonmises.ppf(0.99, B_kappa, loc = B_loc), 100)

# fig, ax = plt.subplots(1, 1)
# # ax.plot(A, vonmises.pdf(A, A_kappa, loc = A_loc), label = 'A')
# # ax.plot(B, vonmises.pdf(B, B_kappa, loc = B_loc), label = 'B')
# A = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
# ax.plot(A, norm.pdf(A))
# plt.legend()
# plt.show()



# # First stance then swing
# stance_ratio = 0.4
# kappa = 8

# def I_stance(phi):
#     return 1 if phi <= stance_ratio else 0


# def I_swing(phi):
#     return 0 if phi <= stance_ratio else 1


# def P_I_stance(phi):
#     if I_stance(phi) == 1:
#         return vonmises.cdf(phi, kappa, stance_ratio) - vonmises.cdf(phi, kappa, 0)
#     else:
#         return 1 - (vonmises.cdf(phi, kappa, stance_ratio) - vonmises.cdf(phi, kappa, 0))

# def P_I_swing(phi):
#     if I_swing(phi) == 1:
#         return vonmises.cdf(phi, kappa, 1) - vonmises.cdf(phi, kappa, stance_ratio)
#     else:
#         return 1 - vonmises.cdf(phi, kappa, 1) - vonmises.cdf(phi, kappa, stance_ratio)


# def expectation(phi):
#     I_st = I_stance(phi)
#     I_sw = I_swing(phi)

#     if I_sw == 1:
#         return -1 * P_I_swing(I_sw)
#     else:
#         return -1 * P_I_stance(I_st)


# fig, ax = plt.subplots(1, 1)
# phi_list = np.linspace(0, 1, 100)
# ax.plot(phi_list, [expectation(phi) for phi in phi_list])


# plt.legend()
# plt.show()





stance_start_mean = -0.5 * 2 * np.pi
stance_end_mean = 0 * 2 * np.pi

next_stance_start_mean = 0.5 * 2 * np.pi
next_stance_end_mean = 1.5 * 2 * np.pi

last_swing_start_mean = -1.5 * 2 * np.pi
last_swing_end_mean = -0.5 * 2 * np.pi

swing_start_mean = 0 * 2 * np.pi
swing_end_mean = 0.5 * 2 * np.pi

kappa = 8


def P_I_stance(phi, x):
    P1 = vonmises_line.cdf(phi, kappa, stance_start_mean)
    P2 = vonmises_line.cdf(phi, kappa, stance_end_mean)

    P3 = vonmises_line.cdf(phi, kappa, next_stance_start_mean)
    P4 = vonmises_line.cdf(phi, kappa, next_stance_end_mean)    

    if x == 1:
        return P1 * (1 - P2) + P3 * (1 - P4)
    else:
        return 1 - (P1 * (1 - P2) + P3 * (1 - P4))


def P_I_swing(phi, x):
    P1 = vonmises_line.cdf(phi, kappa, swing_start_mean)
    P2 = vonmises_line.cdf(phi, kappa, swing_end_mean)

    P3 = vonmises_line.cdf(phi, kappa, last_swing_start_mean)
    P4 = vonmises_line.cdf(phi, kappa, last_swing_end_mean)
    
    if x == 1:
        return P1 * (1 - P2) + P3 * (1 - P4)
    else:
        return 1 - (P1 * (1 - P2) + P3 * (1 - P4))


def expectation_stance(phi):
    return 1 * P_I_stance(phi, 1) + 0 * P_I_stance(phi, 0)


def expectation_swing(phi):
    return 1 * P_I_swing(phi, 1) + 0 * P_I_swing(phi, 0)


# def expectation(phi):
#     if P_I_stance(phi, 1) >= P_I_swing(phi, 1):
#     # if phi <= stance_end_mean:
#         return -1 * expectation_stance(phi)
#     else:
#         return -1 * expectation_swing(phi)
#     # return -1 * expectation_swing(phi) + -1 * expectation_stance(phi)
    


fig, ax = plt.subplots(1, 1)
phi_list = np.linspace(-np.pi, np.pi, 100)
ax.plot(phi_list, [-expectation_swing(phi) for phi in phi_list], label = 'E(I_swing_GRF)')
# ax.plot(phi_list, [-expectation_stance(phi) for phi in phi_list], label = 'E(I_stance_GRF)')
# ax.plot(phi_list, [expectation(phi) for phi in phi_list], label = 'E')

# ax.plot(phi_list, [P_I_stance(phi, 0) for phi in phi_list], label = 'P of I_stance=0')
# ax.plot(phi_list, [P_I_stance(phi, 1) for phi in phi_list], label = 'P of I_stance=1')
# ax.plot(phi_list, [P_I_swing(phi, 0) for phi in phi_list], label = 'P of I_swing=0')
# ax.plot(phi_list, [P_I_swing(phi, 1) for phi in phi_list], label = 'P of I_swing=1')
plt.legend()
plt.show()