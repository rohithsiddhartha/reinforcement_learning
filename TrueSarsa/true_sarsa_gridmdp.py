import numpy as np
import matplotlib.pyplot as plt

'''
blank cell - '0'
obstacle  - 'x'
water - 'w'
goal - 'g'
'''
grid = np.array([['0', '0', '0', '0', '0'],
                 ['0', '0', '0', '0', '0'],
                 ['0', '0', 'x', '0', '0'],
                 ['0', '0', 'x', '0', '0'],
                 ['0', '0', 'w', '0', 'g']])

ACTIONS = ["AR","AD","AL","AU"]

p = {"AU": [ (0.80,(-1,0)), (0.10,(0,0)), (0.05,(0,-1)), (0.05,(0,1)) ],
     "AR": [ (0.80,(0,1)),  (0.10,(0,0)), (0.05,(-1,0)), (0.05,(1,0)) ],
     "AD": [ (0.80,(1,0)),  (0.10,(0,0)), (0.05,(0,1)), (0.05,(0,-1)) ],
     "AL": [ (0.80,(0,-1)), (0.10,(0,0)), (0.05,(1,0)), (0.05,(-1,0)) ]}

WATER = 'w'
OBSTACLE = 'x'
CELL = '0'
GOAL = 'g'

Rewards = { CELL : 0,
            WATER : -10,
            GOAL : 10}

TerminalStates = [(4,4)]

ACTION_DIRS = {"AL": "←",
               "AU": "↑",
               "AR": "→",
               "AD": "↓",
               "  ": " ",
               "G ": "G"}

vf_star = np.array([[4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
                    [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
                    [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
                    [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
                    [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]])

def getAllStates():
    rowlen, collen = grid.shape
    allStates = []
    for row in range(rowlen):
        for col in range(collen):
            allStates.append((row, col))
    allStates.remove((2,2))
    allStates.remove((3,2))
    return allStates

def getInitState():
    # return np.array([0,0])
    states = getAllStates()
    ind = np.random.choice(range(len(states)))
    return states[ind]


def getInitWeight(initW = 20):
    #TODO
    n_states = np.size(grid)
    n_actions = len(ACTIONS)
    # w = np.random.randn(n_states*n_actions)
    w = initW*np.ones(n_states*n_actions)
    return w

def isValidState(s):
    rowlen, collen = grid.shape
    if s[0] < 0 or s[0] >=rowlen or s[1] < 0 or s[1] >= collen:
        return False
    if grid[s[0]][s[1]] == OBSTACLE:
        return False
    return True

# vector pf length s*a size, with one hot encoding
# value corresponding s, a is 1 everything else is 0
def getFeatureVec(s, a):
    # n_states = np.size(grid)
    # n_actions = len(ACTIONS)
    # x = np.zeros(n_states*n_actions)
    # if grid[s[0]][s[1]] == 'g': 
    #     return x
    # x[ACTIONS.index(a)*n_states + grid.shape[0]*s[0] + s[1] ] = 1
    # return x
    rows, cols = grid.shape
    x = np.zeros((len(ACTIONS), rows, cols))
    if tuple(s) in [(4,4), (2,2), (3,2)]:
        x = x.flatten()
        return x
    x[ACTIONS.index(a)][s[0]][s[1]] = 1
    x = x.flatten()
    return x

def getAction(s, w, epsilon):
    q_values = []
    probs = []
    for a in ACTIONS:
        x = getFeatureVec(s, a)
        q_values.append(np.dot(w, x))
    max_inds = list(np.argwhere(q_values == np.max(q_values)).flatten())
    for ind in range(len(ACTIONS)):
        if ind in max_inds:
            probs.append( ((1-epsilon)/len(max_inds)) + epsilon/len(ACTIONS) )
        else:
            probs.append( epsilon/len(ACTIONS) )
    
    a = np.random.choice(ACTIONS,size=1, p=probs)[0]
    return a


def getGreedyAction(s, w):
    q_values = []
    for a in ACTIONS:
        x = getFeatureVec(s, a)
        q_values.append(np.dot(w, x))
    max_inds = list(np.argwhere(q_values == np.max(q_values)).flatten())
    a = ACTIONS[np.random.choice(max_inds)]
    return a


def getNextState(s, a):
    next_states = []
    state_probs = []
    for transition in p[a]:
        prob, smod = transition
        s_next = np.array(s)+np.array(smod)
        if isValidState(s_next):
            next_states.append(s_next)
            state_probs.append(prob)
        else:
            next_states.append(s)
            state_probs.append(prob)
    s_next = next_states[np.random.choice( range(len(next_states)) ,size=1,p=state_probs)[0] ]
    return s_next

def getReward(s_curr, a, s_next):
    if not isValidState(s_next):
        raise Exception("getReward function: not valid next state")
    if grid[s_next[0]][s_next[1]] in Rewards:
        return Rewards[grid[s_next[0]][s_next[1]]]
    raise Exception(f"getReward unknown grid element at {s_next}: {grid[s_next[0]][s_next[1]]}")


def printPolicy(w): 
    print("Greedy Policy:")
    rowlen, collen = grid.shape
    for row in range(rowlen):
        for col in range(collen):
            if grid[row][col] == 'x':
                print(" ", end = "    ")
            elif grid[row][col] == 'g':
                print("G", end = "    ")
            else:    
                print(ACTION_DIRS[getGreedyAction(np.array([row, col]), w)], end = "    ")
                # print(getGreedyAction(np.array([row, col]), w), end = "    ")
        print()
    # print()


def getStateValue(s, w, epsilon):
    q_values = []
    probs = []
    for a in ACTIONS:
        x = getFeatureVec(s, a)
        q_values.append(np.dot(w, x))
    max_inds = list(np.argwhere(q_values == np.max(q_values)).flatten())
    for ind in range(len(ACTIONS)):
        if ind in max_inds:
            probs.append( ((1-epsilon)/len(max_inds)) + epsilon/len(ACTIONS) )
        else:
            probs.append( epsilon/len(ACTIONS) )
    return np.dot(q_values, probs)
    # return np.max(q_values)
    

def getVF(w, epsilon):
    vf = np.zeros(grid.shape)
    states = getAllStates()
    for state in states:
        state = np.array(state)
        vf[state[0]][state[1]] = getStateValue(state, w, epsilon)
    return vf


def getVFMSE(w, epsilon):
    vf = getVF(w, epsilon)
    mse = np.sum((vf-vf_star)**2)/np.size(vf)
    return mse

########################################################


def runTrueOnlineSarsaEpisode(wt, gamma, lamda, alpha, epsilon):
    wt1 = np.copy(wt)
    action_count = 0
    st = getInitState()
    at = getAction(st, wt1, epsilon)
    action_count += 1
    xt = getFeatureVec(st, at)
    z = np.zeros(xt.shape)
    Q_old = 0
    while(True):
        if tuple(st) in TerminalStates:
            break
        st1 = getNextState(st, at)
        r = getReward(st, at, st1)
        at1 = getAction(st1, wt1, epsilon)
        action_count += 1
        xt1 = getFeatureVec(st1, at1)
        Qt = np.dot(wt1, xt)
        Qt1 = np.dot(wt1, xt1)
        delta = r + gamma*Qt1 - Qt
        z = gamma*lamda*z + (1-alpha*gamma*lamda*(np.dot(z, xt)))*xt
        wt1 = wt1 + alpha*(delta + Qt - Q_old)*z - alpha*(Qt - Q_old)*xt
        Q_old = Qt1
        xt = xt1
        st = st1
        at = at1
    return (wt1, action_count)


def runTrueOnlineSarsa(DELTA, gamma, alpha, epsilon, lamda, alpha_red_factor, epsilon_red_factor, initW, DELTA_COUNT):
    wt = getInitWeight(initW)
    nEpisodes = 1
    min_delta = 10000
    action_count_arr = []
    action_count_cumm = 0
    vf_mse_arr = []
    while True:
        wt1, action_count = runTrueOnlineSarsaEpisode(wt, gamma, lamda, alpha, epsilon)
        action_count_cumm += action_count
        action_count_arr.append(action_count_cumm)
        mse = getVFMSE(wt1, epsilon)
        vf_mse_arr.append(mse)

        delta = np.max(abs(wt1 - wt))
        min_delta =  min_delta if delta == 0 else min(min_delta, delta)
        # print(f"episode: {nEpisodes}, min-delta: {min_delta}")
        if delta > 0 and delta < DELTA:
            DELTA_COUNT -= 1
            if DELTA_COUNT == 0:
                break
        
        #TODO
        alpha = alpha - alpha/alpha_red_factor
        epsilon = epsilon - epsilon/epsilon_red_factor
        wt = np.copy(wt1)
        nEpisodes += 1
    return (wt1, nEpisodes, np.array(action_count_arr), np.array(vf_mse_arr), epsilon)


def runTrueOnlineSarsaNtimes(N, DELTA, gamma, alpha, epsilon, lamda, alpha_red_factor, epsilon_red_factor, initW, DELTA_COUNT):
    print(f"Running Sarsa for {N} times")
    print(f"N:{N}, DELTA:{DELTA}, gamma:{gamma}, alpha:{alpha}, epsilon:{epsilon}, lamda:{lamda}, alpha_red_factor:{alpha_red_factor}, epsilon_red_factor:{epsilon_red_factor}, initW:{initW}, DELTA_COUNT:{DELTA_COUNT}")
    wt_mean = np.zeros(np.size(grid)*len(ACTIONS))
    vf_mean = np.zeros(grid.shape)
    n_episodes = None
    action_count_arr_mean = None
    vf_mse_arr_mean = None
    for i in range(N):
        print(f"Running True-Online-Sarsa {i+1}")
        wt, nEps, action_count_arr, vf_mse_arr, epsilon_mod = runTrueOnlineSarsa(DELTA, gamma, alpha, epsilon, lamda, alpha_red_factor, epsilon_red_factor, initW, DELTA_COUNT)
        vf = getVF(wt, epsilon_mod)
        wt_mean += wt
        vf_mean += vf
        if n_episodes is None:
            n_episodes = nEps
            action_count_arr_mean = action_count_arr
            vf_mse_arr_mean = vf_mse_arr
        else:
            n_episodes = min(n_episodes, nEps)
            action_count_arr_mean = action_count_arr_mean[:n_episodes] + action_count_arr[:n_episodes]
            vf_mse_arr_mean = vf_mse_arr_mean[:n_episodes] + vf_mse_arr[:n_episodes]     
    wt_mean = wt_mean/N
    vf_mean = vf_mean/N
    action_count_arr_mean = action_count_arr_mean/N
    vf_mse_arr_mean = vf_mse_arr_mean/N
    
    #=====  part a  ======
    plt.figure()
    plt.title("True Online Sarsa - Total number of actions taken over the episodes")
    plt.xlabel("Total actions taken")
    plt.ylabel("Episode number")
    plt.plot(action_count_arr_mean, list(range(1,n_episodes+1)))
    plt.savefig(f"./SG/SGparta_N{N}_lamda{lamda}.jpg")
    # plt.show()

    #=====  part b  ======
    plt.figure()
    plt.title("True Online Sarsa - VF estimates MSE over the episodes")
    plt.xlabel("Episode number")
    plt.ylabel("current VF estimate MSE")
    plt.plot(list(range(1,n_episodes+1)), vf_mse_arr_mean)
    plt.savefig(f"./SG/SGpartb_N{N}_lamda{lamda}.jpg")
    # plt.show()

    #=====  part c  ======
    #TODO: can we avg wts like this and use it for finding policy
    printPolicy(wt_mean)

    # print(f"wts:\n{wt_mean}")
    print(f"VF:\n{vf_mean}")
    print(f"nEps: {n_episodes}")
    maxnorm = np.max(abs(vf_mean - vf_star))
    print(f"MaxNorm: {maxnorm}")
    return maxnorm

def tuningParameters(N, DELTA, gamma, alpha, epsilon, lamda, alpha_red_factor, epsilon_red_factor, initW, DELTA_COUNT):
    # hyperparameter tuning code
    # maxnorms = []
    # flist = [0.3]
    # feature = "lamda"
    # for lamda in flist:
    #     print(f"\n{feature}: {lamda}")
    #     maxnorm = runTrueOnlineSarsaNtimes(N, DELTA, gamma, alpha, epsilon, lamda, alpha_red_factor, epsilon_red_factor, initW, DELTA_COUNT)
    #     maxnorms.append(maxnorm)
    # # plt.figure()
    # # plt.xlabel(f"{feature}")
    # # plt.ylabel("maxnorm")
    # # plt.plot(flist,maxnorms)
    # # plt.savefig(f"SG_{feature}.jpg")
    # # plt.show()
    pass

def main():
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, suppress=True)

    N = 20
    initW = 20      #[10, 15, 20, 25]
    gamma = 0.9
    lamda = 0.3       #[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]  0.1, 0.3
    DELTA = 0.0001
    DELTA_COUNT = 1
    alpha = 0.1     #[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    epsilon = 0.1   #[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]  0.01
    alpha_red_factor = 1000     #[100, 200, 500, 1000, 1500, 2000, 2500, 5000] 500, 1000, 2000
    epsilon_red_factor = 500   #[100, 200, 500, 1000, 1500, 2000, 2500, 5000] 1000
    
    runTrueOnlineSarsaNtimes(N, DELTA, gamma, alpha, epsilon, lamda, alpha_red_factor, epsilon_red_factor, initW, DELTA_COUNT)


if __name__=="__main__":
    main()