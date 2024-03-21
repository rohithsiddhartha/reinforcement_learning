
import numpy as np
import matplotlib.pyplot as plt
import gym
import itertools


#constants

PI = np.pi

#global variables from gym documentation
cos_t1_min, cos_t1_max = -1, 1
sin_t1_min, sin_t1_max = -1, 1
cos_t2_min, cos_t2_max = -1, 1
sin_t2_min, sin_t2_max = -1, 1
w_t1_min, w_t1_max = -4*PI, 4*PI
w_t2_min, w_t2_max = -4*PI, 9*PI

# Constants for the Acrobot environment
# Three discrete actions: 0 (left), 1 (no force), 2 (right)
# 0: -1 torque, 1: 0 torque, 2: 1 torque
ACTIONS = [0, 1, 2]  

EPISODE_TERMINAL_TIME = 500


def getInitState():
    env = gym.make('Acrobot-v1')
    initial_state, _ = env.reset()
    return initial_state


def getInitWeight(M, initW):
    #TODO
    vec_size =((M+1)**6)
    if initW == -1:
        w = np.random.randn(vec_size)
    else:
        w = initW*np.ones(vec_size)
    return w

def getInitTheta(M):
    return np.random.randn((M+1)**6, len(ACTIONS))


def isTerminalState(s):
    # Check if the Acrobot state is terminal
    return bool(gym.envs.classic_control.acrobot_swingup.AcrobotSwingupEnv.is_terminal(s))

# def getFeatureVec(s, M, use_cos=True):
#     # vec_size = ((M + 1) ** 4)
#     # if isTerminalState(s):
#     #     return np.zeros(vec_size)

#     cos_t1, sin_t1, cos_t2, sin_t2, w_t1, w_t2 = s

#     global cos_t1_min, cos_t1_max, sin_t1_min, sin_t1_max
#     global cos_t2_min, cos_t2_max, sin_t2_min, sin_t2_max
#     global w_t1_min, w_t1_max, w_t2_min, w_t2_max

#     cos_t1 = (cos_t1 - cos_t1_min) / (cos_t1_max - cos_t1_min)
#     sin_t1 = (sin_t1 - sin_t1_min) / (sin_t1_max - sin_t1_min)
#     cos_t2 = (cos_t2 - cos_t2_min) / (cos_t2_max - cos_t2_min)
#     sin_t2 = (sin_t2 - sin_t2_min) / (sin_t2_max - sin_t2_min)
#     w_t1 = (w_t1 - w_t1_min) / (w_t1_max - w_t1_min)
#     w_t2 = (w_t2 - w_t2_min) / (w_t2_max - w_t2_min)

#     xf_main = []
#     if use_cos:
#         for i in range(M + 1):
#             for j in range(M + 1):
#                 for k in range(M + 1):
#                     for l in range(M + 1):
#                         for m in range(M + 1):
#                             for n in range(M + 1):
#                                 xf_main.append(
#                                     i * cos_t1 + j * sin_t1 + k * cos_t2 + l * sin_t2 + m * w_t1 + n * w_t2
#                                 )
#         xf_main = np.array(xf_main)
#         xf_main = np.cos(PI * xf_main)
#     else:
#         for i in range(M + 1):
#             for j in range(M + 1):
#                 for k in range(M + 1):
#                     for l in range(M + 1):
#                         for m in range(M + 1):
#                             for n in range(M + 1):
#                                 xf_main.append(
#                                     i * cos_t1 + j * sin_t1 + k * cos_t2 + l * sin_t2 + m * w_t1 + n * w_t2
#                                 )
#         xf_main = np.array(xf_main)
#         xf_main = np.sin(PI * xf_main)

#     return xf_main


def getFeatureVec(s, M, use_cos=True):
    cos_t1, sin_t1, cos_t2, sin_t2, w_t1, w_t2 = s

    cos_t1 = (cos_t1 - cos_t1_min) / (cos_t1_max - cos_t1_min)
    sin_t1 = (sin_t1 - sin_t1_min) / (sin_t1_max - sin_t1_min)
    cos_t2 = (cos_t2 - cos_t2_min) / (cos_t2_max - cos_t2_min)
    sin_t2 = (sin_t2 - sin_t2_min) / (sin_t2_max - sin_t2_min)
    w_t1 = (w_t1 - w_t1_min) / (w_t1_max - w_t1_min)
    w_t2 = (w_t2 - w_t2_min) / (w_t2_max - w_t2_min)

    # Create a grid of indices for all combinations of i, j, k, l, m, n
    # i, j, k, l, m, n = np.meshgrid(range(M + 1), range(M + 1), range(M + 1), range(M + 1), range(M + 1), range(M + 1))
    
    # # Calculate feature values using vectorized operations
    # xf_main = (
    #     i * cos_t1 + j * sin_t1 + k * cos_t2 + l * sin_t2 + m * w_t1 + n * w_t2
    # )
    

    indices = np.indices((M + 1, M + 1, M + 1, M + 1, M + 1, M + 1)).reshape(6, -1)
    
    # Unpack indices for clarity
    i, j, k, l, m, n = indices

    # Calculate feature values using vectorized operations
    xf_main = (
        i * cos_t1 + j * sin_t1 + k * cos_t2 + l * sin_t2 + m * w_t1 + n * w_t2
    )
    if use_cos:
        xf_main = np.cos(PI * xf_main)
    else:
        xf_main = np.sin(PI * xf_main)

    # Flatten the grid and return the result
    return xf_main.flatten()


def getNextState(s, a):
    # Apply the selected action to transition to the next state in Acrobot
    env = gym.make('Acrobot-v1')
    env.reset()
    env.env.state = np.array(s)
    next_state, _, terminal_status, _ , _ = env.step(a)
    return tuple(next_state), terminal_status

def getReward(s_curr, a, s_next):
    # Get the reward for the Acrobot environment
    env = gym.make('Acrobot-v1')
    env.reset()
    env.env.state = np.array(s_curr)
    _, reward, _, _, _ = env.step(a)
    return reward


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def value(s, w, M, use_cos=True):
    features = getFeatureVec(s, M, use_cos)
    return np.dot(features, w)


def getAction(s, theta, M, use_cos=True):

    phi_s = getFeatureVec(s, M, use_cos)
    
    # Compute the policy's outputs for each action
    policy_outputs = np.dot(theta.T, phi_s)
    
    # Apply softmax to determine action probabilities
    action_probs = softmax(policy_outputs)
    
    # Select an action based on the probabilities
    action = np.random.choice(len(ACTIONS), p=action_probs)
    
    return action, action_probs


def runEpisode(theta, w, alpha_theta, alpha_w, gamma, M, use_cos=True):
    s = getInitState()  # Initialize S (first state of episode)
    I = 1.0  # Initialize I to 1 for the first step of each episode
    total_reward = 0
    timestep = 0
    action_count=0

    while True:
        timestep+=1

        if timestep > EPISODE_TERMINAL_TIME:
            break
        # Choose action A based on current policy
        phi_s = getFeatureVec(s, M)

        a, probs = getAction(s, theta, M, use_cos)

        action_count+=1

        # Take action A, observe S', R
        s_next, terminal_check = getNextState(s, ACTIONS[a])

        r = getReward(s, ACTIONS[a], s_next)
        # total_reward+=r
        
        # Compute TD error (delta)
        if terminal_check==True:
            delta = r - value(s, w, M, use_cos)  # If S' is terminal, then v̂(S',w) is 0
        else:
            delta = r + gamma * value(s_next, w, M, use_cos) - value(s, w, M, use_cos)
        
        
        # Compute gradients
        w += alpha_w * delta * phi_s

        for i, prob in enumerate(probs):

            if i == a:
                theta[:, i] += alpha_theta * I * delta * phi_s * (1 - probs[a])
            else:
                theta[:, i] -= alpha_theta * I * delta * phi_s * probs[a]
        

        # Update I
        I *= gamma

        # Transition to new state
        s = s_next
        if terminal_check==True:
            break

        #Checking reward and restricting time
    print("Reward:", timestep)
    return theta, w, action_count 

def estimate_J(theta, M, Jeps):
    #here we only run one episode, because every episode will give same output
    #because we are taking actions greedily wrt w

    # alpha_theta, alpha_w are not necessary as we aren't updating weights here
    # we are checking the rewards for this specific weight and theta trained

    #w is used in state value calculation which isn't necessary here
    G_arr = []
    nEpisodes = Jeps
    for _ in range(nEpisodes):
        s = getInitState()
        timestep = 0
        G = 0
        # terminal_check = False
        while timestep < EPISODE_TERMINAL_TIME:
            #TODO: greedy action 
            _, at_probs = getAction(s, theta, M)
            at = np.argmax(at_probs)

            s_next, terminal_check = getNextState(s, ACTIONS[at])
            r = getReward(s, at, s_next)
            G = G + r
            s = s_next
            timestep += 1
            if terminal_check==True:
                break
        G_arr.append(G)
    # print(G_arr[-1])
    return np.mean(G_arr)


def run_OAC(nTrial ,nIters, M, gamma, theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara=""):   
    w = getInitWeight(M, initW)
    theta = getInitTheta(M)

    J_iters = []
    action_count_iters = []
    action_count_cumm = 0
    for iter in range(1, nIters+1):
        print(f"Trial: {nTrial}, iter - {iter}/{nIters}", end = " ")
        theta_t, w, action_count =  runEpisode(theta, w, alpha_theta, alpha_w, gamma, M)

        action_count_cumm += action_count
        action_count_iters.append(action_count_cumm)
        ret = estimate_J(theta_t, M, Jeps)
        # print(ret)
        J_iters.append(ret)
        # check=200
        if iter>alpha_theta_red_factor:
            alpha_w = alpha_w*alpha_w_red_factor/iter
            alpha_theta = alpha_theta*alpha_theta_red_factor/iter
            # alpha_theta = max(0, alpha_theta - alpha_theta/alpha_red_factor)
            # alpha_w = max(0, alpha_w - alpha_w/alpha_w_red_factor)

    return (np.array(J_iters), np.array(action_count_iters) )



def runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara=""):
    J_trials = []
    action_count_trials = []
    for i in range(N):
        print(f"OAC running: {i+1}/{N}")
        J_iter, action_count_iter = run_OAC(f"{i+1}/{N}", nIters, M, gamma,theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW)
        J_trials.append(J_iter)
        action_count_trials.append(action_count_iter)
    J_trials = np.array(J_trials)
    action_count_trials = np.array(action_count_trials)
    mean_J = np.mean(J_trials, axis=0)
    std_J = np.std(J_trials, axis=0)
    mean_action_count = np.mean(action_count_trials, axis=0)

    plt.figure()
    plt.title("Total number of actions taken over the episodes")
    plt.xlabel("Total actions taken")
    plt.ylabel("Episode number")
    plt.plot(mean_action_count, list(range(1,nIters+1)) )
    plt.savefig(f"./tuning/AC/{hpara}/SC_AE_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}_{alpha_theta_red_factor}_{alpha_w_red_factor}).jpg")
    plt.close()
    
    plt.figure()
    plt.title("Mean Return vs iteration number")
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), mean_J)
    plt.savefig(f"./tuning/AC/{hpara}/SC_Mean_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}.jpg")
    plt.show(block=False)
    plt.pause(1)
    plt.close()


    plt.figure()
    plt.title("Mean, Standard deviation of Returns vs iteration number")
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.errorbar(list(range(1,nIters+1)), mean_J, std_J, linestyle='-', ecolor='lightsteelblue')
    plt.savefig(f"./tuning/AC/{hpara}/SC_MStd_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}_{alpha_theta_red_factor}_{alpha_w_red_factor}).jpg")
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.title("Standard deviation of Returns vs iteration number")
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), std_J)
    plt.savefig(f"./tuning/AC/{hpara}/SC_Std_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}_{alpha_theta_red_factor}_{alpha_w_red_factor}).jpg")
    plt.close()

    


def tuningParameters(N, nIters, M, gamma, alpha_w, alpha_theta, use_cos, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW):

    s_init = getInitState()
    feat = getFeatureVec(s_init, M)
    w = np.random.randn((M+1)**6)
    print("***"*10)
    print(feat.shape, w.shape)
    print("***"*10)

    theta = np.random.randn((M+1)**6, len(ACTIONS))


    # finding alpha_theta - stepsize
    print("Finding alpha_theta...")
    feat_arr = [5e-4, 1e-5, 5e-5,0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    for alpha_theta in feat_arr:
        print(f"alpha_theta: {alpha_theta}:")
        runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="alpha_theta")

    # # finding alpha_w - learning rate for w
    # print("Finding alpha_w...")
    # feat_arr = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.9]
    # for alpha_w in feat_arr:
    #     print(f"alpha_w: {alpha_w}:")
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="alpha_w")
    

    # # finding alpha_theta_red_factor
    # print("Finding alpha_theta_red_factor...")
    # feat_arr = [100, 500, 1000, 1500, 2000]
    # for alpha_theta_red_factor in feat_arr:
    #     print(f"alpha_theta_red_factor: {alpha_theta_red_factor}:")
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara= "alphaRed")
    

    # # finding alpha_w_red_factor 
    # print("Finding alpha_w_red_factor...")
    # feat_arr = [100, 500, 1000, 1500, 2000]
    # for alpha_w_red_factor in feat_arr:
    #     print(f"alpha_w_red_factor: {alpha_w_red_factor}:")
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="epsilonRed")
    
    
    # finding M 
    # print("Finding M...")
    # feat_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    # for M in feat_arr:
    #     print(f"M: {M}:")
    #     w = np.random.randn((M+1)**6)
    #     theta = np.random.randn((M+1)**6, len(ACTIONS))
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="M")
    

    # # finding gamma
    # print("Finding gamma...")
    # feat_arr = [0.9, 0.92, 0.94, 0.96, 0.98, 1]
    # for gamma in feat_arr:
    #     print(f"gamma: {gamma}:")
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="gamma")
    # return


def main():
    # np.random.seed(7) 
    N = 1
    nIters = 500
    Jeps = 1
    initW = -1
    gamma = 1
    M = 3
    use_cos = True
    alpha_theta = 1e-3
    alpha_w = 5e-4
    alpha_theta_red_factor = 200
    alpha_w_red_factor = 200   

    s_init = getInitState()
    feat = getFeatureVec(s_init, M)
    w = np.random.randn((M+1)**6)
    theta = np.random.randn((M+1)**6, len(ACTIONS))


    # tuningParameters(N, nIters, M, gamma, alpha_w, alpha_theta, use_cos, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW)
    runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="final_seed7")


if __name__=="__main__":
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    main()