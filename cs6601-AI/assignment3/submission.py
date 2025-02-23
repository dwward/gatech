import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from random import randint
import numpy


# You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # p_temp_gauge_working = 0.95
    # p_temp_gauge_faulty = 0.20
    # p_alarm_faulty = 0.15
    # p_temp_hot = 0.20
    # p_temp_gauge_faulty_hot_temp = 0.80
    # p_temp_gauge_faulty_normal_temp = 0.05
    # p_alarm_responds_to_gauge = 0.90
    # p_alarm_responds_to_gauge = 0.55

    '''
          TEMP --------+
           |           |
           v           v
         GAUGE <--- FAULTY GAUGE
           |
           V
         ALARM <--- FAULTY ALARM
    '''
    BayesNet = BayesianModel()

    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")

    BayesNet.add_edge("temperature", "gauge")
    BayesNet.add_edge("temperature", "faulty gauge")
    BayesNet.add_edge("gauge", "alarm")
    BayesNet.add_edge("faulty gauge", "gauge")
    BayesNet.add_edge("faulty alarm", "alarm")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature".
    (for the tests to work.)
    """

    # Temperature (1)
    #    The temperature is hot (call this "true") 20% of the time.
    #       (high = True, normal = False)
    # p(temp)
    #
    cpd_temp = TabularCPD('temperature', 2, values=[[0.80], [0.20]])  # F, T

    # Gauge (4)
    #     The temperature gauge reads the correct temperature with 95% probability when it is not faulty and
    #     20% probability when it is faulty. For simplicity, say that the gauge's "true" value corresponds
    #     with its "hot" reading and "false" with its "normal" reading, so the gauge would have a 95% chance
    #     of returning "true" when the temperature is hot and it is not faulty.
    #       Temperature (high/hot = True, normal = False)
    #
    # p(gauge|temp, faulty gauge)
    #
    cpd_gauge = TabularCPD('gauge', 2, values=[[0.95, 0.20, 0.05, 0.80],  # p(gauge|temp, faulty gauge)
                                               [0.05, 0.80, 0.95, 0.20]],  # FF, FT, TF, TT
                           # cpd_gauge = TabularCPD('gauge', 2, values=[[0.95, 0.20, 0.05, 0.80],  # p(gauge|temp, faulty gauge)
                           #                                            [0.05, 0.80, 0.95, 0.20]],  # FF, FT, TF, TT

                           evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])

    # Faulty Gauge (2)
    #     When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% 
    #     of the time.
    # p(faulty gauge|temp)
    #
    cpd_faulty_gauge = TabularCPD('faulty gauge', 2, values=[[0.95, 0.20],
                                                             [0.05, 0.80]],  # F, T
                                  evidence=['temperature'], evidence_card=[2])

    # Alarm (4)
    #     The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, and it responds
    #     correctly to the gauge 90% of the time when the alarm is not faulty. For instance, when it is faulty,
    #     the alarm sounds 55% of the time that the gauge is "hot" and remains silent 55% of the time that the
    #     gauge is "normal."
    #       Gauge Hot (high/hot = True, normal = False)
    # p(alarm|gauge, faulty alarm)
    #
    cpd_alarm = TabularCPD('alarm', 2, values=[[0.90, 0.55, 0.10, 0.45],  # p(alarm|gauge, faulty alarm)
                                               [0.10, 0.45, 0.90, 0.55]],  # FF, FT, TF, TT
                           evidence=['gauge', 'faulty alarm'], evidence_card=[2, 2])

    # cpd_alarm = TabularCPD('alarm', 2, values=[[0.90, 0.55, 0.10, 0.45],  # p(alarm|gauge, faulty alarm)
    #                                            [0.10, 0.45, 0.90, 0.55]],  # FF, FT, TF, TT
    #                        evidence=['gauge', 'faulty alarm'], evidence_card=[2, 2])

    # Faulty Alarm (1)
    #     The alarm is faulty 15% of the time.
    # p(faulty alarm)
    #
    cpd_faulty_alarm = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])

    bayes_net.add_cpds(cpd_temp, cpd_gauge, cpd_faulty_gauge, cpd_alarm, cpd_faulty_alarm)

    return bayes_net


def sanity_check(bayes_net):
    """Calculate the marginal probability of the alarm ringing in the power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['temperature'], joint=False)
    temp_prob = marginal_prob['temperature'].values
    return temp_prob[1]


def get_alarm_prob(bayes_net):
    """Calculate the marginal probability of the alarm ringing in the power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values
    return alarm_prob[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal probability of the gauge showing hot in the power plant system."""
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    prob = marginal_prob['gauge'].values
    gauge_prob = prob
    return gauge_prob[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability of the temperature being hot in the power plant system,
    given that the alarm sounds and neither the gauge nor alarm is faulty."""
    solver = VariableElimination(bayes_net)                       ### this... double check
    conditional_prob = solver.query(variables=['temperature'], evidence={'alarm': 1, 'faulty alarm': 0, 'faulty gauge': 0},
                                    joint=False)
    prob = conditional_prob['temperature'].values

    temp_prob = prob
    return temp_prob[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    bayes_net = BayesianModel()
    bayes_net.add_nodes_from(["A", "B", "C", "AvB", "BvC", "CvA"])
    bayes_net.add_edges_from([("A", "AvB"), ("A", "CvA"), ("B", "AvB"), ("B", "BvC"), ("C", "BvC"), ("C", "CvA")])

    skill_levels_dist = [[0.15], [0.45], [0.30], [0.10]]

    versus_dist = [
        #  0    T2+1  T2+2  T2+3     T1+1    0   T2+1  T2+2    T1+2   T1+1   0    T2+1    T1+3   T1+2  T1+1    0
        [0.10, 0.20, 0.15, 0.05] + [0.60, 0.10, 0.20, 0.15] + [0.75, 0.60, 0.10, 0.20] + [0.90, 0.75, 0.60, 0.10],  # T1
        [0.10, 0.60, 0.75, 0.90] + [0.20, 0.10, 0.60, 0.75] + [0.15, 0.20, 0.10, 0.60] + [0.05, 0.15, 0.20, 0.10],  # T2
        [0.80, 0.20, 0.10, 0.05] + [0.20, 0.80, 0.20, 0.10] + [0.10, 0.20, 0.80, 0.20] + [0.05, 0.10, 0.20, 0.80],  # T
    ]

    cpd_A = TabularCPD('A', 4, values=skill_levels_dist)
    cpd_B = TabularCPD('B', 4, values=skill_levels_dist)
    cpd_C = TabularCPD('C', 4, values=skill_levels_dist)

    # [cols, rows1, rows2]
    cpd_AvB = TabularCPD('AvB', 3, values=versus_dist, evidence=['A', 'B'], evidence_card=[4, 4])
    cpd_BvC = TabularCPD('BvC', 3, values=versus_dist, evidence=['B', 'C'], evidence_card=[4, 4])
    cpd_CvA = TabularCPD('CvA', 3, values=versus_dist, evidence=['C', 'A'], evidence_card=[4, 4])

    bayes_net.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)

    # bayes_net.check_model()

    return bayes_net


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], evidence={'AvB': 0, 'CvA': 2},
                                    joint=False)

    # print(bayes_net.get_cpds("BvC"))
    posterior = conditional_prob['BvC'].values
    return posterior  # list


def Gibbs_sampler(bayes_net, initial_state, target_index=None):
    skill_range = [0, 1, 2, 3]
    outcome_range = [0, 1, 2]  # [b wins, b loses, b ties]

    team_table = bayes_net.get_cpds('B').values
    match_table = bayes_net.get_cpds("BvC").values

    index_map = {
        "A": 0, "B": 1, "C": 2, "AvB": 3, "BvC": 4, "CvA": 5,
        0: "A", 1: "B", 2: "C", 3: "AvB", 4: "BvC", 5: "CvA"
    }
    new_state = numpy.asarray(initial_state)

    def rand_skill():
        return randint(0, 3)
        # return numpy.random.choice(skill_range, p=team_table)

    def rand_outcome():
        return randint(0, 2)

    if not initial_state:
        new_state = numpy.asarray([rand_skill(), rand_skill(), rand_skill(), 0, rand_outcome(), 2])

    mutable_index = randint(0, 3)  # a skill, b skill, c skill, BvC
    if target_index is not None:
        target_index = target_index

    # children = bayes_net.get_children(index_map[mutable_index])
    # parents = bayes_net.get_children(index_map[mutable_index])

    skill_a = new_state[0]
    skill_b = new_state[1]
    skill_c = new_state[2]
    outcome_avb = new_state[3]
    outcome_bvc = new_state[4]
    outcome_cva = new_state[5]

    if mutable_index == 3:  # Calculate BvC value
        # Calculate P(BvC|mb(BvC) == P(BvC|B,C)
        index_BvC = 4
        p_match_dist = match_table[:, skill_b, skill_c]
        new_state[index_BvC] = numpy.random.choice(outcome_range, p=p_match_dist)

    elif mutable_index <= 2:  # Index 0,1,2 are skills for A,B,C

        p_skill_dist = numpy.zeros(4)
        normalizer = 0

        # Calculate P(Xi|mb(Xi))
        # P(xi|parents(xi)) * pi (xi's children y | y...parents)
        # markov blanket: Parents, Children, Children Parents,
        # The skill nodes have no parents, skill nodes have children, the children have no other parents
        # .... for BvC
        #   p(B) * P(BvC|B,C) * P(AvB|A,B)
        #
        for i in range(0, 4):
            p_ch1 = None
            p_ch2 = None
            if mutable_index == 0:  # A
                skill_a = i
                p_curr_skill = team_table[skill_a]
                p_ch1 = match_table[outcome_avb, skill_a, skill_b]
                p_ch2 = match_table[outcome_cva, skill_c, skill_a]
            elif mutable_index == 1:  # B
                skill_b = i
                p_curr_skill = team_table[skill_b]
                p_ch1 = match_table[outcome_avb, skill_a, skill_b]
                p_ch2 = match_table[outcome_bvc, skill_b, skill_c]
            elif mutable_index == 2:  # C
                skill_c = i
                p_curr_skill = team_table[skill_c]
                p_ch1 = match_table[outcome_bvc, skill_b, skill_c]
                p_ch2 = match_table[outcome_cva, skill_c, skill_a]
            else:
                print("something wrong")

            p = p_curr_skill * p_ch1 * p_ch2
            p_skill_dist[i] = p
            normalizer = normalizer + p

        p_skill_dist = p_skill_dist / normalizer
        new_state[mutable_index] = numpy.random.choice(skill_range, p=p_skill_dist)

    else:
        print("This definitely ain't right")

    sample = tuple(new_state)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """
    skill_range = [0, 1, 2, 3]
    outcome_range = [0, 1, 2]  # For BvC [b wins, b loses, b ties]

    team_table = bayes_net.get_cpds('B').values
    match_table = bayes_net.get_cpds("BvC").values

    index_map = {
        "A": 0, "B": 1, "C": 2, "AvB": 3, "BvC": 4, "CvA": 5,
        0: "A", 1: "B", 2: "C", 3: "AvB", 4: "BvC", 5: "CvA"
    }

    def rand_skill():
        return numpy.random.choice(skill_range)

    def rand_outcome(s1, s2):  # might need to calculate this over real distribution.... here and in gibbs.  nope.
        return numpy.random.choice(outcome_range)

    def random_state():
        s1 = rand_skill()  # a
        s2 = rand_skill()  # b
        s3 = rand_skill()  # c
        o1 = rand_outcome(s2, s3)
        return numpy.asarray([s1, s2, s3, 0, o1, 2])

    def p_state(tmp_state):
        p_A = team_table[tmp_state[0]]
        p_B = team_table[tmp_state[1]]
        p_C = team_table[tmp_state[2]]
        p_AvB = match_table[0, tmp_state[0], tmp_state[1]]  # fixed e
        p_BvC = match_table[tmp_state[4], tmp_state[1], tmp_state[2]]
        p_CvA = match_table[2, tmp_state[2], tmp_state[0]]  # fixed e
        p_proposed = p_A * p_B * p_C * p_AvB * p_BvC * p_CvA
        return p_proposed

    if not initial_state:
        return random_state()

    u = numpy.random.uniform(0.0, 1.0)
    proposed_state = random_state()
    p_proposed = p_state(proposed_state)
    p_previous = p_state(initial_state)
    accept = min(1, p_proposed / p_previous)

    # accept new proposal only if it is an improvement, and less than a random u
    if u < accept:
        new_state = proposed_state
    else:
        new_state = initial_state

    # new_state_list = new_state.tolist()
    sample = tuple(new_state)
    return sample


def gibbs_converge(game_network, initial_state):
    # ------------------------------
    # GIBBS SAMPLING
    # ------------------------------
    gibbs_result = Gibbs_sampler(game_network, initial_state)
    gibbs_converge = numpy.asarray([0, 0, 0])
    for i in range(0, 10000):  # burn, baby burn!  A few throwaways before counting, hopefully converges better
        gibbs_result = Gibbs_sampler(game_network, list(gibbs_result))
    last_t1 = None
    delt = 0.0001
    N = 1000000  # Convergence iterations
    N_count = 0

    Gibbs_count = 0
    while True:
        gibbs_result = Gibbs_sampler(game_network, list(gibbs_result))
        Gibbs_count = Gibbs_count + 1
        bvc_result = gibbs_result[4]
        # We want [win, loss, tie]

        if bvc_result == 0:  # B WIN
            gibbs_converge[0] = gibbs_converge[0] + 1
        elif bvc_result == 1:  # B LOSS
            gibbs_converge[1] = gibbs_converge[1] + 1
        elif bvc_result == 2:  # BvC TIE
            gibbs_converge[2] = gibbs_converge[2] + 1

        if last_t1 is not None:
            curr_p_t1 = gibbs_converge[0] / Gibbs_count
            diff = abs(curr_p_t1 - last_t1)
            last_t1 = curr_p_t1
            # print(diff)
            if diff < delt:
                N_count = N_count + 1
            else:
                N_count = 0
        else:
            last_t1 = gibbs_converge[0]

        if N_count == N:
            break

    # stationary distribution p(x1...xn|e)
    gibbs_converge = numpy.divide(gibbs_converge, Gibbs_count)
    return gibbs_converge, Gibbs_count


def mh_converge(game_network, initial_state):
    # ------------------------------
    # METROPOLIS-HASTINGS SAMPLING
    # ------------------------------
    mh_result = MH_sampler(game_network, initial_state)
    mh_converge = numpy.asarray([0, 0, 0])
    for i in range(0, 10000):  # burn, baby burn!  A few throwaways before counting, hopefully converges better
        mh_result = MH_sampler(game_network, list(mh_result))
    last_t1 = None
    delt = 0.0001
    N = 1000000  # Convergence iterations
    N_count = 0
    MH_count = 0
    MH_reject = 0

    while True:
        last_result = mh_result
        mh_result = MH_sampler(game_network, list(mh_result))

        if mh_result == last_result:
            MH_reject = MH_reject + 1
        MH_count = MH_count + 1
        bvc_result = mh_result[4]

        if bvc_result == 0:  # B WIN
            mh_converge[0] = mh_converge[0] + 1
        elif bvc_result == 1:  # B LOSS
            mh_converge[1] = mh_converge[1] + 1
        elif bvc_result == 2:  # BvC TIE
            mh_converge[2] = mh_converge[2] + 1

        if last_t1 is not None:
            curr_p_t1 = mh_converge[0] / MH_count
            diff = abs(curr_p_t1 - last_t1)
            last_t1 = curr_p_t1
            # print(diff)
            if diff < delt:
                N_count = N_count + 1
            else:
                N_count = 0
        else:
            last_t1 = mh_converge[0]

        if N_count == N:
            break

    # stationary distribution p(x1...xn|e)
    mh_converge = numpy.divide(mh_converge, MH_count)
    return mh_converge, MH_count, MH_reject


def compare_sampling(bayes_net, initial_state, mode='b'):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = None  # [0, 0, 0]  # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = None  # [0, 0, 0]  # posterior distribution of the BvC match as produced by MH

    # game_network = get_game_network()

    # [win, loss, tie]
    # engine_posterior = calculate_posterior(bayes_net)
    # print("Engine calculated posterior {}".format(engine_posterior))
    print("Posterior from pgmpy: Engine calculated posterior [0.25890074 0.42796763 0.31313163]")

    if mode == 'g' or mode == 'b':
        # Gibbs
        print("Running Gibbs")
        Gibbs_convergence, Gibbs_count = gibbs_converge(bayes_net, initial_state)
        print("Gibbs: {} Gibbs_Count: {}".format(Gibbs_convergence, Gibbs_count))
    if mode == 'm' or mode == 'b':
        # MH
        print("Running Metro-Hastings")
        MH_convergence, MH_count, MH_rejection_count = mh_converge(bayes_net, initial_state)
        print("Convergence: {} MH_Count: {} Rejects: {}".format(MH_convergence, MH_count, MH_rejection_count))

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    choice = 0
    options = ['Gibbs', 'Metropolis-Hastings']
    factor = 1.79
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return 'dward45'
