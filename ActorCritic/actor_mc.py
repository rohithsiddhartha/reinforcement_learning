import numpy as np
import matplotlib.pyplot as plt



#constants
f = 0.001
g = 0.0025
PI = np.pi
R = -1
EPISODE_TERMINAL_TIME = 200

#global variables
x_min, x_max = -1.2, 0.6
v_min, v_max = -0.07, 0.07

# left, dont move, right
ACTIONS = [0, 1, 2]


def getInitState():
    x = np.random.choice(np.arange(-0.6, -0.3, 0.1))
    v = 0
    return (x,v)


def getInitWeight(M, initW):
    vec_size = ((M+1)**2)
    if initW == -1:
        w = np.random.randn(vec_size)
    else:
        w = initW*np.ones(vec_size)
    return w

def getInitTheta(M):
    return np.random.randn((M+1)**2, len(ACTIONS))


def getReward(s_curr, a, s_next):
    return R

def isTerminalState(st):
    x, _ = st
    if x >= 0.5:
        return True
    return False


def getFeatureVec(st, M, use_cos=True):
    
    xt, vt = st
    xf_main = []
    if use_cos:
        x = (xt - x_min)/(x_max - x_min)
        v = (vt - v_min)/(v_max - v_min)
        for i in range(M+1):
            for j in range(M+1):
                xf_main.append(i*x + j*v)
        xf_main = np.array(xf_main)
        xf_main = np.cos(PI*xf_main)
    else:
        x = 2*(xt - x_min)/(x_max - x_min) - 1
        v = 2*(vt - v_min)/(v_max - v_min) - 1
        for i in range(M+1):
            for j in range(M+1):
                xf_main.append(i*x + j*v)
        xf_main = np.array(xf_main)
        xf_main = np.sin(PI*xf_main)

    return xf_main


def getNextState(s, a):
    xt, vt = s
    vt1 = vt + (a-1)*f - (np.cos(3*xt))*g
    xt1 = xt + vt1
    if xt1 < x_min:
        xt1 = x_min
        vt1 = 0
    elif xt1 > x_max:
        xt1 = x_max
        vt1 = 0
    vt1 = min(vt1, v_max) if vt1 >=0 else max(vt1, v_min)
    return (xt1, vt1)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def value(s, w, M, use_cos=True):
    phi_s = getFeatureVec(s, M, use_cos)
    return np.dot(phi_s, w)

def getAction(st, theta, M, use_cos=True):

    phi_s = getFeatureVec(st, M, use_cos)
    
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
    # w = np.copy(wt)
    while not isTerminalState(s):

        timestep+=1

        if timestep > EPISODE_TERMINAL_TIME:
            break
        # Choose action A based on current policy
        phi_s = getFeatureVec(s, M)

        a, probs = getAction(s, theta, M)

        action_count+=1

        # Take action A, observe S', R
        
        s_next = getNextState(s, a)

        # phi_s_next = getFeatureVec(s_next, M)

        r = getReward(s, a, s_next)
        
        # Compute TD error (delta)
        if isTerminalState(s_next):
            delta = r - value(s, w, M, use_cos)  # If S' is terminal, then v(S',w) is 0
        else:
            delta = r + gamma * value(s_next, w, M, use_cos) - value(s, w, M, use_cos)
        
        # Compute gradients
        w = w + alpha_w * delta * phi_s

        for i, prob in enumerate(probs):
            # print(i, prob)

            if i == a:
                theta[:, i] += alpha_theta * I * delta * phi_s * (1 - probs[a])
            else:
                theta[:, i] -= alpha_theta * I * delta * phi_s * probs[a]

        # Update I
        I *= gamma
        # Transition to new state
        s = s_next

    print("Reward",timestep)
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
        while timestep < EPISODE_TERMINAL_TIME:
            if isTerminalState(s):
                break
            #TODO: greedy action or normal action?
            _, action_probs = getAction(s, theta, M)
            max_prob_actions = np.where(action_probs == np.max(action_probs))[0]
            at = np.random.choice(max_prob_actions)

            s_next = getNextState(s, at)
            r = getReward(s, at, s_next)
            G = G + r
            s = s_next
            timestep += 1
        G_arr.append(G)
    return np.mean(G_arr)


def run_OAC(nTrial ,nIters, M, gamma, theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara=""):   
    w = getInitWeight(M, initW)
    theta = getInitTheta(M)

    J_iters = []
    action_count_iters = []
    action_count_cumm = 0
    for iter in range(1, nIters+1):
        # print("run_OAC happening")
        print(f"Trial: {nTrial}, iter - {iter}/{nIters}", end = " ")
        theta, w, action_count =  runEpisode(theta, w, alpha_theta, alpha_w, gamma, M)
        action_count_cumm += action_count
        action_count_iters.append(action_count_cumm)
        J_iters.append(estimate_J(theta, M, Jeps))
        if iter>min(alpha_theta_red_factor, alpha_w_red_factor):
            if iter>alpha_w_red_factor:
                alpha_w = alpha_w*alpha_w_red_factor/iter
            if iter>alpha_theta_red_factor:
                alpha_theta = alpha_theta*alpha_theta_red_factor/iter


    return ( np.array(J_iters), np.array(action_count_iters) )


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
    plt.suptitle("Total number of actions taken over the episodes")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("Total actions taken")
    plt.ylabel("Episode number")
    plt.plot(mean_action_count, list(range(1,nIters+1)) )
    plt.savefig(f"./tuning/MC/{hpara}/SC_AE_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}).jpg")
    plt.close()
    
    plt.figure()
    plt.suptitle("Mean Return vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), mean_J)
    plt.savefig(f"./tuning/MC/{hpara}/SC_Mean_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}.jpg")
    plt.show(block=False)
    plt.pause(1)
    plt.close()


    plt.figure()
    plt.suptitle("Mean, Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.errorbar(list(range(1,nIters+1)), mean_J, std_J, linestyle='-', ecolor='lightsteelblue')
    plt.savefig(f"./tuning/MC/{hpara}/SC_MStd_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}).jpg")
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.suptitle("Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), std_J)
    plt.savefig(f"./tuning/MC/{hpara}/SC_Std_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}).jpg")
    plt.close()
    

def tuningParameters(N, nIters, M, gamma, alpha_w, alpha_theta, use_cos, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW):

    # N = 10
    # nIters = 10
    # Jeps = 1
    # # best values found
    # alpha_theta = 0.001
    # alpha_w = 0.01
    # alpha_theta_red_factor = 1000
    # alpha_w_red_factor = 1500  #dont use alpha_w reduction at all
    # M = 4
    # gamma = 0.9
    # s_init = getInitState()
    # feat = getFeatureVec(s_init, M)
    w = np.random.randn((M+1)**2)
    # print("***"*10)
    # print(feat.shape, w.shape)
    # print("***"*10)

    theta = np.random.randn((M+1)**2, len(ACTIONS))


    # finding alpha_theta - stepsize
    print("Finding alpha_theta...")
    # feat_arr = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    # for alpha_theta in feat_arr:
    #     print(f"alpha_theta: {alpha_theta}:")
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="alpha_theta")

    # # finding alpha_w - learning rate for w
    # print("Finding alpha_w...")
    feat_arr = [0.1, 0.05] # 0.01, 0.1, 0.2, 0.5, 0.9]
    for alpha_w in feat_arr:
        print(f"alpha_w: {alpha_w}:")
        runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="alpha_w")
    
        
    # finding M 
    # print("Finding M...")
    # feat_arr = [4, 5, 6, 7, 8, 9, 10, 15, 20]
    # for M in feat_arr:
    #     print(f"M: {M}:")
    #     s_init = getInitState()
    #     feat = getFeatureVec(s_init, M)
    #     w = np.random.randn(feat.shape[0])

    #     theta = np.random.randn((M+1)**2, len(ACTIONS))
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="M")
    

    # # finding gamma
    # print("Finding gamma...")
    # feat_arr = [0.9, 0.92, 0.94, 0.96, 0.98, 1]
    # for gamma in feat_arr:
    #     print(f"gamma: {gamma}:")
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="gamma")

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
    

    # return


def main():
    N = 10
    nIters = 5000
    Jeps = 5
    initW = -1
    gamma = 1
    M = 5
    use_cos = True
    alpha_theta = 0.2
    alpha_w = 0.001
    # alpha_theta = 0.001
    # alpha_w = 0.001
    alpha_theta_red_factor = 1250
    alpha_w_red_factor = 1500   

    s_init = getInitState()
    feat = getFeatureVec(s_init, M)
    w = np.random.randn((M+1)**2)

    theta = np.random.randn((M+1)**2, len(ACTIONS))


    # tuningParameters(N, nIters, M, gamma, alpha_w, alpha_theta, use_cos, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW)
    runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="final_seed7")

    # Ms = [4, 5, 6, 10]

    # # count = 0
 
    # for M in Ms:
    #     print(f"M: {M}")
    #     w = np.random.randn((M+1)**2)
    #     theta = np.random.randn((M+1)**2, len(ACTIONS))
    #     runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara=f"final_run_cum_M")
        


if __name__=="__main__":
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    main()
