import numpy as np
import matplotlib.pyplot as plt


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
    vec_size = len(ACTIONS)*((M+1)**4)
    if initW == -1:
        w = np.random.randn(vec_size)
    else:
        w = initW*np.ones(vec_size)
    return w


def isTerminalState(s):
    x_t, _, w_t, _ = s
    if x_t < -2.4 or x_t > 2.4:
        return True
    if w_t < -PI/15 or w_t > PI/15:
        return True
    return False

def getFeatureVecOld(s, a, M, use_cos):
    if isTerminalState(s):
        return np.zeros(8*M+2)
    
    x_t, v_t, w_t, wh_t = s
    global vt_min
    global vt_max
    global wht_max
    global wht_min
    vt_min = min(vt_min, v_t)
    vt_max = max(vt_max, v_t)
    wht_min = min(wht_min, wh_t)
    wht_max = max(wht_max, wh_t)

    if use_cos:
        x = (x_t + 2.4)/4.8
        v = (v_t - vt_min)/(vt_max - vt_min)
        w = (w_t + PI/15)/(2*PI/15)
        wh = (wh_t - wht_min)/(wht_max - wht_min)
        x_main = np.cos(np.concatenate(( np.array([0]), np.arange(1, M+1)*PI*x, np.arange(1, M+1)*PI*v, np.arange(1, M+1)*PI*w, np.arange(1, M+1)*PI*wh )))
    else:
        x = x_t/2.4
        v = 2*(v_t - vt_min)/(vt_max - vt_min) - 1
        w = w_t/(PI/15)
        wh = 2*(wh_t - wht_min)/(wht_max - wht_min) - 1
        x_main = np.sin(np.concatenate(( np.array([PI/2]), np.arange(1, M+1)*PI*x, np.arange(1, M+1)*PI*v, np.arange(1, M+1)*PI*w, np.arange(1, M+1)*PI*wh )))

    x_rest = np.zeros(4*M+1)
    if a == "left":
        x = np.concatenate((x_main, x_rest))
    else:
        x = np.concatenate((x_rest, x_main))
    return x

def getFeatureVec(s, a, M, use_cos):
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


    xf_rest = np.zeros((M+1)**4)
    if a == "left":
        xf = np.concatenate((xf_main, xf_rest))
    else:
        xf = np.concatenate((xf_rest, xf_main))
    return xf
    

def getAction(s, w, epsilon, M, use_cos):
    q_values = []
    probs = []
    for a in ACTIONS:
        x = getFeatureVec(s, a, M, use_cos)
        q_values.append(np.dot(w, x))
    max_inds = list(np.argwhere(q_values == np.max(q_values)).flatten())
    for ind in range(len(ACTIONS)):
        if ind in max_inds:
            probs.append( ((1-epsilon)/len(max_inds)) + epsilon/len(ACTIONS) )
        else:
            probs.append( epsilon/len(ACTIONS) )
    a = np.random.choice(ACTIONS,1, p=probs)[0]
    return a

def getGreedyAction(s, w, epsilon, M, use_cos):
    q_values = []
    probs = []
    for a in ACTIONS:
        x = getFeatureVec(s, a, M, use_cos)
        q_values.append(np.dot(w, x))
    max_inds = list(np.argwhere(q_values == np.max(q_values)).flatten())
    a = ACTIONS[np.random.choice(max_inds)]
    return a


def getNextState(s, a):
    x_t, v_t, w_t, wh_t = s
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




########################################################



def runEpisode(wt, gamma, lamda, alpha, epsilon, M, use_cos):
    action_count = 0
    wt1 = np.copy(wt)
    st = getInitState()
    at = getAction(st, wt1, epsilon, M, use_cos)
    action_count += 1
    xt = getFeatureVec(st, at, M, use_cos)
    z = np.zeros(xt.shape)
    Q_old = 0
    timestep = 0
    while timestep < 500:
        if isTerminalState(st):
            break
        st1 = getNextState(st, at)
        r = getReward(st, at, st1)
        at1 = getAction(st1, wt1, epsilon, M, use_cos)
        action_count += 1
        xt1 = getFeatureVec(st1, at1, M, use_cos)
        Qt = np.dot(wt1, xt)
        Qt1 = np.dot(wt1, xt1)
        delta = r + gamma*Qt1 - Qt
        z = (gamma*lamda)*z + (1-(alpha*gamma*lamda)*(np.dot(z, xt)))*xt
        # print(f"=========================\ntime:{timestep}, a:{alpha},d:{delta},qt:{Qt},qtold:{Q_old}, maxwt:{np.max(wt1)}, maxz:{np.max(z)}\nz:{z},\nxt:{xt}\nwt1:{wt1}")
        wt1 = wt1 + (alpha*(delta + Qt - Q_old))*z - (alpha*(Qt - Q_old))*xt
        # print(f"wt1:{wt1}")
        Q_old = Qt1
        xt = xt1
        st = st1
        at = at1
        timestep += 1
    print(f"timestep: {timestep}")
    return (wt1, action_count)

def estimate_J(w, gamma, epsilon, M, use_cos, Jeps):
    #here we only run one episode, because every episode will give same output
    #because we are taking actions greedily wrt w
    G_arr = []
    nEpisodes = Jeps
    for i in range(nEpisodes):
        eps = 0
        st = getInitState()
        timestep = 0
        G = 0
        while timestep < 500:
            #TODO: greedy action or normal action?
            at = getGreedyAction(st, w, eps, M, use_cos)
            st1 = getNextState(st, at)
            r = getReward(st, at, st1)
            G = G + r
            st = st1
            timestep += 1
            if isTerminalState(st):
                break
        G_arr.append(G)
    return np.mean(G_arr)

def runSarsa(nTrial ,nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, hpara=""):   
    wt = getInitWeight(M, initW)
    J_iters = []
    action_count_iters = []
    action_count_cumm = 0
    for iter in range(1, nIters+1):
        print(f"Trial: {nTrial}, iter - {iter}/{nIters}", end = " ")
        wt1, action_count =  runEpisode(wt, gamma, lamda, alpha, epsilon, M, use_cos)
        action_count_cumm += action_count
        action_count_iters.append(action_count_cumm)
        J_iters.append(estimate_J(wt1, gamma, epsilon, M, use_cos, Jeps))
        wt = np.copy(wt1)
        alpha = max(0, alpha - alpha/alpha_red_factor)
        # epsilon = max(0, epsilon - epsilon/epsilon_red_factor)

    return ( np.array(J_iters), np.array(action_count_iters) )

def runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, hpara=""):
    J_trials = []
    action_count_trials = []
    for i in range(N):
        print(f"Sarsa running: {i+1}/{N}")
        J_iter, action_count_iter = runSarsa(f"{i+1}/{N}", nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW)
        J_trials.append(J_iter)
        action_count_trials.append(action_count_iter)
    J_trials = np.array(J_trials)
    action_count_trials = np.array(action_count_trials)
    mean_J = np.mean(J_trials, axis=0)
    std_J = np.std(J_trials, axis=0)
    mean_action_count = np.mean(action_count_trials, axis=0)
    
    plt.figure()
    plt.suptitle("Mean Return vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, α:{alpha}, eps:{epsilon}, α-red:{alpha_red_factor}, eps-red:{epsilon_red_factor}, lamda:{lamda})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), mean_J)
    plt.savefig(f"./SC/SC_Mean_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    plt.figure()
    plt.suptitle("Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, α:{alpha}, eps:{epsilon}, α-red:{alpha_red_factor}, eps-red:{epsilon_red_factor}, lamda:{lamda})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), std_J)
    plt.savefig(f"./SC/SC_Std_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.close()

    plt.figure()
    plt.suptitle("Mean, Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, α:{alpha}, eps:{epsilon}, α-red:{alpha_red_factor}, eps-red:{epsilon_red_factor}, lamda:{lamda})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.errorbar(list(range(1,nIters+1)), mean_J, std_J, linestyle='-', ecolor='lightsteelblue')
    plt.savefig(f"./SC/SC_MStd_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.suptitle("Total number of actions taken over the episodes")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, α:{alpha}, eps:{epsilon}, α-red:{alpha_red_factor}, eps-red:{epsilon_red_factor}, lamda:{lamda})", fontsize = 9)
    plt.xlabel("Total actions taken")
    plt.ylabel("Episode number")
    plt.plot(mean_action_count, list(range(1,nIters+1)) )
    plt.savefig(f"./SC/SC_AE_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.close()



def tuningParameters(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW):
    N = 1
    nIters = 150
    Jeps = 1
    # best values found
    alpha = 0.001
    epsilon = 0.05
    alpha_red_factor = 1000
    epsilon_red_factor = 1500  #dont use epsilon reduction at all
    M = 4
    lamda = 0.4
    gamma = 1


    # # finding alpha - stepsize
    # print("Finding alpha...")
    # feat_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    # for alpha in feat_arr:
    #     print(f"alpha: {alpha}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "alpha")
    

    # # finding epsilon - exploration parameter
    # print("Finding epsilon...")
    # feat_arr = [ 0.1, 0.2, 0.5, 0.9]
    # for epsilon in feat_arr:
    #     print(f"epsilon: {epsilon}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "epsilon")
    

    # # finding alpha_red_factor
    # print("Finding alpha_red_factor...")
    # feat_arr = [100, 500, 1000, 1500, 2000]
    # for alpha_red_factor in feat_arr:
    #     print(f"alpha_red_factor: {alpha_red_factor}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "alphaRed")
    

    # # finding epsilon_red_factor 
    # print("Finding epsilon_red_factor...")
    # feat_arr = [100, 500, 1000, 1500, 2000]
    # for epsilon_red_factor in feat_arr:
    #     print(f"epsilon_red_factor: {epsilon_red_factor}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "epsilonRed")
    
    
    # # finding M 
    # print("Finding M...")
    # feat_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    # for M in feat_arr:
    #     print(f"M: {M}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "M")
    

    # # finding lamda
    # print("Finding lamda...")
    # feat_arr = [0.1, 0.2, 0.3, 0.4, 0.5]
    # for lamda in feat_arr:
    #     print(f"lamda: {lamda}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "lamda")


    # # finding gamma
    # print("Finding gamma...")
    # feat_arr = [0.9, 0.92, 0.94, 0.96, 0.98, 1]
    # for gamma in feat_arr:
    #     print(f"gamma: {gamma}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "gamma")
    # return


def main():
    np.random.seed(7) 
    N = 10
    nIters = 150
    Jeps = 5
    initW = -1
    gamma = 1
    lamda = 0.2  #0.4
    M = 4
    use_cos = True
    alpha = 0.001
    epsilon = 0.05
    alpha_red_factor = 1000
    epsilon_red_factor = 1500   #dont use epsilon reduction at all


    # tuningParameters(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW)
    runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "final_seed7")


if __name__=="__main__":
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    main()