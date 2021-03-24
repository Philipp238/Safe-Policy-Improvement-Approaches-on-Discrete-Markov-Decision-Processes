# import time
# import numpy as np
#
# from batch_rl_algorithms.basic_rl import Basic_rl
# from batch_rl_algorithms.pi_star import Pi_star
# from batch_rl_algorithms.spibb import SPIBB
# from batch_rl_algorithms.spibb import Lower_SPIBB
# from batch_rl_algorithms.r_min import RMin
# from batch_rl_algorithms.soft_spibb import ApproxSoftSPIBB
# from batch_rl_algorithms.soft_spibb import ExactSoftSPIBB
# from batch_rl_algorithms.soft_spibb import LowerApproxSoftSPIBB
# from batch_rl_algorithms.soft_spibb import AdvantageousApproxSoftSPIBB
# from batch_rl_algorithms.duipi import DUIPI
# from batch_rl_algorithms.ramdp import RaMDP
# from batch_rl_algorithms.mbie import MBIE
#
#
# def basic_rl(pi_b, gamma, nb_states, nb_actions, data, R_state_state, episodic, R_state_action, P, results,
#              env_parameters):
#     rl_new = Basic_rl(pi_b=pi_b, gamma=gamma, nb_states=nb_states, nb_actions=nb_actions, data=data,
#                       R=R_state_state, episodic=episodic)
#     t_0 = time.time()
#     rl_new.fit()
#     t_1 = time.time()
#     pi_rl_perf = policy_evaluation_exact(rl_new.pi, R_state_action, P, gamma)[0][0]
#     print(f'Pi_rl_new_perf: {pi_rl_perf}')
#     to_append = env_parameters + []
#     results.append(to_append)
#
#
# def policy_evaluation_exact(pi, r, p, gamma):
#     """
#     Evaluate policy by taking the inverse
#     Args:
#       pi: policy, array of shape |S| x |A|
#       r: the true rewards, array of shape |S| x |A|
#       p: the true state transition probabilities, array of shape |S| x |A| x |S|
#     Return:
#       v: 1D array with updated state values
#     """
#     # Rewards according to policy: Hadamard product and row-wise sum
#     r_pi = np.einsum('ij,ij->i', pi, r)
#
#     # Policy-weighted transitions:
#     # multiply p by pi by broadcasting pi, then sum second axis
#     # result is an array of shape |S| x |S|
#     p_pi = np.einsum('ijk, ij->ik', p, pi)
#     # v = np.dot(np.linalg.inv((np.eye(p_pi.shape[0]) - gamma * p_pi)), r_pi)
#     # New calculation to make it more stable
#     v = np.linalg.solve((np.eye(p_pi.shape[0]) - gamma * p_pi), r_pi)
#     return v, r + gamma * np.einsum('i, jki->jk', v, p)
