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


def getInitWeightOld(M, initW):
    vec_size = (2*M+1)*3
    if initW == -1:
        w = np.random.randn(vec_size)
    else:
        w = initW*np.ones(vec_size)
    return w

def getInitWeight(M, initW):
    #TODO
    vec_size = len(ACTIONS)*((M+1)**2)
    if initW == -1:
        w = np.random.randn(vec_size)
    else:
        w = initW*np.ones(vec_size)
    return w


def isTerminalState(s):
    x, _ = s
    if x >= 0.5:
        return True
    return False


def getFeatureVecOld(s, a, M, use_cos):
    vec_size = (2*M+1)*3
    if isTerminalState(s):
        return np.zeros(vec_size)
    
    xt, vt = s
    if use_cos:
        x = (xt - x_min)/(x_max - x_min)
        v = (vt - v_min)/(v_max - v_min)
        xf_main = np.cos(np.concatenate(( np.array([0]), np.arange(1, M+1)*PI*x, np.arange(1, M+1)*PI*v )))
    else:
        x = 2*(xt - x_min)/(x_max - x_min) - 1
        v = 2*(vt - v_min)/(v_max - v_min) - 1
        xf_main = np.sin(np.concatenate(( np.array([PI/2]), np.arange(1, M+1)*PI*x, np.arange(1, M+1)*PI*v )))

    xf_rest1 = np.zeros(2*M+1)
    xf_rest2 = np.zeros(2*M+1)
    if a == 0:
        xf = np.concatenate((xf_main, xf_rest1, xf_rest2))
    elif a == 1:
        xf = np.concatenate((xf_rest1, xf_main, xf_rest2))
    else:
        xf = np.concatenate((xf_rest1, xf_rest2, xf_main))
    return xf

def getFeatureVec(s, a, M, use_cos):
    #TODO
    vec_size = len(ACTIONS)*((M+1)**2)
    if isTerminalState(s):
        return np.zeros(vec_size)
    
    xt, vt = s
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

    xf_rest1 = np.zeros((M+1)**2)
    xf_rest2 = np.zeros((M+1)**2)
    if a == 0:
        xf = np.concatenate((xf_main, xf_rest1, xf_rest2))
    elif a == 1:
        xf = np.concatenate((xf_rest1, xf_main, xf_rest2))
    else:
        xf = np.concatenate((xf_rest1, xf_rest2, xf_main))
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


def getGreedyAction(s, w, M, use_cos):
    q_values = []
    for a in ACTIONS:
        x = getFeatureVec(s, a, M, use_cos)
        q_values.append(np.dot(w, x))
    max_inds = list(np.argwhere(q_values == np.max(q_values)).flatten())
    a = ACTIONS[np.random.choice(max_inds)]
    return a


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
    while timestep < EPISODE_TERMINAL_TIME:
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
        Q_old = Qt1
        # print(f"time: {timestep}, x: {st[0]}")
        # print(f"\n\ntimestep: {timestep}\nst: {st}, at:{at}, st1:{st1}, at1:{at1}, r:{r}\nxt:{xt}\nwt:{wt1}")
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
        while timestep < EPISODE_TERMINAL_TIME:
            if isTerminalState(st):
                break
            #TODO: greedy action or normal action?
            at = getGreedyAction(st, w, M, use_cos)
            st1 = getNextState(st, at)
            r = getReward(st, at, st1)
            G = G + r
            st = st1
            timestep += 1
        G_arr.append(G)
    return np.mean(G_arr)

def runSarsa(nTrial, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, hpara=""):
    wt = getInitWeight(M, initW)
    J_iters = []
    action_count_iters = []
    action_count_cumm = 0
    for iter in range(1, nIters+1):
        print(f"Trial:{nTrial}, iter - {iter}/{nIters}", end = " ")
        wt1, action_count =  runEpisode(wt, gamma, lamda, alpha, epsilon, M, use_cos)
        action_count_cumm += action_count
        action_count_iters.append(action_count_cumm)
        J = estimate_J(wt1, gamma, epsilon, M, use_cos, Jeps)
        J_iters.append(J)
        wt = np.copy(wt1)
        alpha = alpha - alpha/alpha_red_factor
        # epsilon = epsilon - epsilon/epsilon_red_factor
    
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
    plt.savefig(f"./SM/SM_Mean_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    plt.figure()
    plt.suptitle("Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, α:{alpha}, eps:{epsilon}, α-red:{alpha_red_factor}, eps-red:{epsilon_red_factor}, lamda:{lamda})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.plot(list(range(1,nIters+1)), std_J)
    plt.savefig(f"./SM/SM_Std_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.close()

    plt.figure()
    plt.suptitle("Mean, Standard deviation of Returns vs iteration number")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, α:{alpha}, eps:{epsilon}, α-red:{alpha_red_factor}, eps-red:{epsilon_red_factor}, lamda:{lamda})", fontsize = 9)
    plt.xlabel("iteration/update number")
    plt.ylabel("Return")
    plt.errorbar(list(range(1,nIters+1)), mean_J, std_J, linestyle='-', ecolor='lightsteelblue')
    plt.savefig(f"./SM/SM_MStd_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.show(block=False)
    plt.close()

    plt.figure()
    plt.suptitle("Total number of actions taken over the episodes")
    plt.title(f"(N:{N}, nIters:{nIters}, Jeps:{Jeps}, M:{M}, gamma:{gamma}, α:{alpha}, eps:{epsilon}, α-red:{alpha_red_factor}, eps-red:{epsilon_red_factor}, lamda:{lamda})", fontsize = 9)
    plt.xlabel("Total actions taken")
    plt.ylabel("Episode number")
    plt.plot(mean_action_count, list(range(1,nIters+1)) )
    plt.savefig(f"./SM/SM_AE_{hpara}_({N}_{nIters}_{Jeps}_{M}_{gamma}_{alpha}_{epsilon}_{alpha_red_factor}_{epsilon_red_factor}_{lamda}).jpg")
    plt.close()



def tuningParameters(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, hpara):
    N = 5
    nIters = 3000
    Jeps = 3
    # best values found
    alpha = 0.01
    epsilon = 0.05
    # alpha_red_factor = 1000
    # epsilon_red_factor = 1500  #dont use epsilon reduction at all
    # M = 4
    # lamda = 0.4
    # gamma = 1


    # # finding alpha - stepsize
    # print("Finding alpha...")
    # feat_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
    # for alpha in feat_arr:
    #     print(f"alpha: {alpha}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "alpha")
    

    # finding epsilon - exploration parameter
    # print("Finding epsilon...")
    # feat_arr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
    # for epsilon in feat_arr:
    #     print(f"epsilon: {epsilon}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "epsilon")
    

    # # finding alpha_red_factor
    # print("Finding alpha_red_factor...")
    # feat_arr = [100, 500, 1000, 1500, 2000, 2500]
    # for alpha_red_factor in feat_arr:
    #     print(f"alpha_red_factor: {alpha_red_factor}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "alphaRed")

    # # finding epsilon_red_factor 
    # print("Finding epsilon_red_factor...")
    # feat_arr = [100, 500, 1000, 1500, 2000, 2500]
    # for epsilon_red_factor in feat_arr:
    #     print(f"epsilon_red_factor: {epsilon_red_factor}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "epsilonRed")
    
    
    # # finding M 
    # print("Finding M...")
    # feat_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20]
    # for M in feat_arr:
    #     print(f"M: {M}:")
    #     runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "M")
    return

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
    nIters = 3000
    Jeps = 3
    initW = -1
    gamma = 1
    lamda = 0
    M = 10
    use_cos = True
    alpha = 0.005
    epsilon = 0.005
    alpha_red_factor = 2000
    epsilon_red_factor = 1000

    tuningParameters(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "")
    # runSarsaNtimes(N, nIters, M, gamma, epsilon, lamda, alpha, use_cos, Jeps, alpha_red_factor, epsilon_red_factor, initW, "")
    

if __name__=="__main__":
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)
    main()