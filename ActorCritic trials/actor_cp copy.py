
import numpy as np
import matplotlib.pyplot as plt
import gym

'''
representation of variables:
state at time t - (x_t,v_t,w_t,wh_t)
state at tinme t+1 - (x_tp1, v_tp1, w_tp1, wh_tp1)
phi_s - numpy array
'''
#constants
g = 9.8
mc = 1.0
mp = 0.1
mt = mc + mp
l = 0.5
tau = 0.02
R = 1
gamma = 1.0
PI = np.pi

#global variables
vt_min, vt_max = -6, 6
wht_min, wht_max = -4, 4

ACTIONS = ["left","right"]


def getInitState():
    return (0,0,0,0)

def getInitWeightOld(M, initW):
    if initW == -1:
        w = np.random.randn(8*M + 2)
    else:
        w = initW*np.ones(8*M + 2)
    return w

def getInitWeight(M, initW):
    #TODO
    vec_size =((M+1)**4)
    if initW == -1:
        w = np.random.randn(vec_size)
    else:
        w = initW*np.ones(vec_size)
    return w


def isTerminalState(s):
    # print(s)
    x_t, _, w_t, _ = s
    if x_t < -2.4 or x_t > 2.4:
        return True
    if w_t < -PI/15 or w_t > PI/15:
        return True
    return False


def getFeatureVec(s, M, use_cos=True):
    vec_size = len(ACTIONS)*((M+1)**4)
    if isTerminalState(s):
        return np.zeros(vec_size)
    
    x_t, v_t, w_t, wh_t = s
    global vt_min
    global vt_max
    global wht_max
    global wht_min
    vt_min = min(vt_min, v_t)
    vt_max = max(vt_max, v_t)
    wht_min = min(wht_min, wh_t)
    wht_max = max(wht_max, wh_t)
    
    xf_main = []
    if use_cos:
        x = (x_t + 2.4)/4.8
        v = (v_t - vt_min)/(vt_max - vt_min)
        w = (w_t + PI/15)/(2*PI/15)
        wh = (wh_t - wht_min)/(wht_max - wht_min)
        for i in range(M+1):
            for j in range(M+1):
                for k in range(M+1):
                    for l in range(M+1):
                        xf_main.append(i*x + j*v + k*w + l*wh)
        xf_main = np.array(xf_main)
        xf_main = np.cos(PI*xf_main)
    else:
        x = x_t/2.4
        v = 2*(v_t - vt_min)/(vt_max - vt_min) - 1
        w = w_t/(PI/15)
        wh = 2*(wh_t - wht_min)/(wht_max - wht_min) - 1
        for i in range(M+1):
            for j in range(M+1):
                for k in range(M+1):
                    for l in range(M+1):
                        xf_main.append(i*x + j*v + k*w + l*wh)
        xf_main = np.array(xf_main)
        xf_main = np.sin(PI*xf_main)

    return xf_main




def getNextState(s, a):
    x_t, v_t, w_t, wh_t = s
    # print(a)
    # verbal_action = ACTIONS[a]
    F = -10 if a == "left" else 10
    
    b = ( F + mp*l*wh_t*wh_t*np.sin(w_t) )/mt
    c = ( g*np.sin(w_t) - b*np.cos(w_t) )/ (l*(4/3 - (mp*np.cos(w_t*w_t)/mt)))
    d = b - ((mp*l*c*np.cos(w_t))/mt)

    x_t1 = x_t + tau*v_t
    v_t1 = v_t + tau*d
    w_t1 = w_t + tau*wh_t
    wh_t1 = wh_t + tau*c
    return (x_t1, v_t1, w_t1, wh_t1)

def getReward(s_curr, a, s_next):
    return R


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def value(s, w, M, use_cos=True):
    features = getFeatureVec(s, M, use_cos)
    return np.dot(features, w)

def policy(s, theta, M, use_cos=True):
    features = getFeatureVec(s, M, use_cos)
    z = np.dot(features, theta)
    return softmax(z)

def get_action(s, theta, M, use_cos=True):
    probs = policy(s, theta, M, use_cos)
    actions = np.random.choice(len(ACTIONS), p=probs)
    return actions, probs


def getAction(s, theta, M, use_cos=True):

    phi_s = getFeatureVec(s, M, use_cos)
    
    # Compute the policy's outputs for each action
    policy_outputs = np.dot(theta.T, phi_s)
    # policy_outputs = np.dot(phi_s, theta)
    
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
    # print(s)
    while not isTerminalState(s):
        timestep+=1

        if timestep > 500:
            break
        # Choose action A based on current policy
        phi_s = getFeatureVec(s, M)
        a, probs = getAction(s, theta, M, use_cos)
        # print(a)
        action_count+=1
        # Take action A, observe S', R
        s_next = getNextState(s, ACTIONS[a])
        # phi_s_next = getFeatureVec(s_next, M)

        r = getReward(s, ACTIONS[a], s_next)
        # total_reward+=r
        
        # Compute TD error (delta)
        if isTerminalState(s_next):
            delta = r - value(s, w, M, use_cos)  # If S' is terminal, then v̂(S',w) is 0
        else:
            delta = r + gamma * value(s_next, w, M, use_cos) - value(s, w, M, use_cos)
        
        
        # Compute gradients
        w += alpha_w * delta * phi_s

        # grad_v = features  # Gradient of value function with respect to w is the feature vector itself
        # grad_log_pi = features * (1 - probs[a])  # Gradient of log policy with respect to theta

        # # Update critic by gradient descent
        
        # # Update actor by gradient ascent
        theta[:, a] += alpha_theta * I * delta * phi_s * (1-probs[a])
        theta[:, 1-a] -= alpha_theta * I * delta * phi_s* probs[a]
        # print(a, probs)
        # for i, prob in enumerate(probs):
        #     print(i, prob)

        #     if i == a:
        #         theta[:, i] += alpha_theta * I * delta * phi_s * (1 - prob)
        #     else:
        #         theta[:, i] -= alpha_theta * I * delta * phi_s * prob
        # break

        # Update I
        I *= gamma
        # Transition to new state
        s = s_next

        #Checking reward and restricting time
    print("Reward:", timestep)
    return theta, w, action_count #, total_reward

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
        while timestep < 500:
            #TODO: greedy action 
            _, at_probs = getAction(s, theta, M)
            # print(at_probs)
            at = np.argmax(at_probs)
            # print(_, at)

            s_next = getNextState(s, ACTIONS[at])
            r = getReward(s, at, s_next)
            G = G + r
            s = s_next
            timestep += 1
            if isTerminalState(s_next):
                break
        G_arr.append(G)
    # print(G_arr[-1])
    return np.mean(G_arr)


def run_OAC(nTrial ,nIters, M, gamma, theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara=""):   
    # w = getInitWeight(M, initW)
    w = np.random.randn((M+1)**4)
    theta = np.random.randn((M+1)**4, len(ACTIONS))

    J_iters = []
    action_count_iters = []
    action_count_cumm = 0
    for iter in range(1, nIters+1):
        print(f"Trial: {nTrial}, iter - {iter}/{nIters}", end = " ")
        theta_t, wt, action_count =  runEpisode(theta, w, alpha_theta, alpha_w, gamma, M)
        if wt.all()==w.all():
            print("SAME")
        else:
            break
        action_count_cumm += action_count
        action_count_iters.append(action_count_cumm)
        ret = estimate_J(theta_t, M, Jeps)
        print(ret)
        J_iters.append(ret)
        # check=100
        # if iter>check:
        #     alpha_w = alpha_w*check/iter
        #     alpha_theta = alpha_theta*check/iter
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
    plt.suptitle("Total number of actions taken over the episodes")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("Total actions taken")
    plt.ylabel("Episode number")
    plt.plot(mean_action_count, list(range(1,nIters+1)) )
    plt.savefig(f"./tuning/CP/{hpara}/SC_AE_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}_{alpha_theta_red_factor}_{alpha_w_red_factor}).jpg")
    plt.close()
    
    plt.figure()
    plt.suptitle("Mean Return vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), mean_J)
    plt.savefig(f"./tuning/CP/{hpara}/SC_Mean_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}.jpg")
    plt.show(block=False)
    plt.pause(1)
    plt.close()


    plt.figure()
    plt.suptitle("Mean, Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.errorbar(list(range(1,nIters+1)), mean_J, std_J, linestyle='-', ecolor='lightsteelblue')
    plt.savefig(f"./tuning/CP/{hpara}/SC_MStd_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}_{alpha_theta_red_factor}_{alpha_w_red_factor}).jpg")
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.suptitle("Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, alpha_theta:{alpha_theta}, alpha_w:{alpha_w})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), std_J)
    plt.savefig(f"./tuning/CP/{hpara}/SC_Std_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha_theta}_{alpha_w}_{alpha_theta_red_factor}_{alpha_w_red_factor}).jpg")
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
    s_init = getInitState()
    feat = getFeatureVec(s_init, M)
    w = np.random.randn((M+1)**4)
    print("***"*10)
    print(feat.shape, w.shape)
    print("***"*10)

    theta = np.random.randn((M+1)**4, len(ACTIONS))


    # finding alpha_theta - stepsize
    print("Finding alpha_theta...")
    feat_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
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
    #     w = np.random.randn(feat.shape[0])
    #     theta = np.random.randn((M+1)**4, len(ACTIONS))
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
    nIters = 150
    Jeps = 1
    initW = -1
    gamma = 1
    M = 4
    use_cos = True
    alpha_theta = 1e-4
    # alpha_w = 4e-4 #best so far or finalized
    alpha_w = 4e-4
    # alpha_theta = 0.001
    # alpha_w = 0.001
    alpha_theta_red_factor = 1000
    alpha_w_red_factor = 1500   #dont use alpha_w reduction at all

    s_init = getInitState()
    feat = getFeatureVec(s_init, M)
    w = np.random.randn(feat.shape[0])
    theta = np.random.randn((M+1)**4, len(ACTIONS))


    tuningParameters(N, nIters, M, gamma, alpha_w, alpha_theta, use_cos, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW)
    # runOAC_Ntimes(N, nIters, M, gamma , theta, w, alpha_theta, alpha_w, Jeps, alpha_theta_red_factor, alpha_w_red_factor, initW, hpara="final_seed7")


if __name__=="__main__":
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    main()