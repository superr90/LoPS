import pandas as pd
import numpy as np
from BasicStrategy.DataPreProcessHuman import human_data_preprocess
from BasicStrategy.GetUtilityHuman import get_human_utility
from BasicStrategy.CorrectUtilityHuman import correct_human_utility
from BasicStrategy.FittingWeightHuman import get_human_weight
from BasicStrategy.reviseWeight.reviseHuman import correct_human_weight
from BasicStrategy.GetUtilityMonkey import get_monkey_utility
from BasicStrategy.CorrectUtilityMonkey import correct_monkey_utility
from BasicStrategy.FittingWeightMonkey import get_monkey_weight
from BasicStrategy.reviseWeight.reviseMonkey import correct_monkey_weight

from FeatureExtractor.ExtractFeaturesHuman import extract_human_feature
from FeatureExtractor.ExtractFeaturesMonkey import extract_monkey_feature

from GrammarInduction.DataFormedHuman import form_date_human
from GrammarInduction.DataFormedMonkey import form_date_monkey

from GrammarInduction.LearnStateHuman import learm_state_graph_human
from GrammarInduction.LearnStateMonkey import learm_state_graph_monkey

from GrammarInduction.GrammarInductionHuman import grammar_induction_human
from GrammarInduction.GrammarInductionMonkey import grammar_induction_monkey

from GrammarInduction.GenerateGramDepthHuman import LoPS_complexity_human
from GrammarInduction.GenerateGramDepthMonkey import LoPSC_omplexity_monkey

from GrammarInduction.GrammarProcess import DividePerson, GrammarAlign

from FeatureExtractor.PerformanceHuman import reaction_time_human, reward_human
from FeatureExtractor.PerformanceMonkey import reaction_time_monkey, reward_monkey
from DrawImg.generate_data.GenerateData import *
from DrawImg.generate_data.Fig6 import *

from DrawImg.plot_code.plot_main import *


def generate_data():
    dates = ["session1", "session2"]
    for date in dates:
        # human data preprocess
        human_data_preprocess(date)
        # get utility of every strategy
        get_human_utility(date)
        correct_human_utility(date)
        # get weight of every strategy
        get_human_weight(date)
        correct_human_weight(date)
        # extract feature
        extract_human_feature(date)
        # form data
        form_date_human(date)
        # learn state
        learm_state_graph_human(date)
        # grammar induction
        grammar_induction_human(date)
        DividePerson(date)
        GrammarAlign("human", date)
        # LoPS Complexity
        LoPS_complexity_human(date)

        # Performance
        reaction_time_human(date)
        reward_human(date)

    dates = ["Year1", "Year2", "Year3"]
    for date in dates:
        # get utility of every strategy
        get_monkey_utility(date)
        correct_monkey_utility(date)
        # get weight of every strategy
        get_monkey_weight(date)
        correct_monkey_weight(date)
        # extract feature
        extract_monkey_feature(date)

        # form data
        form_date_monkey(date)
        # learn state
        learm_state_graph_monkey("Omega", date)
        learm_state_graph_monkey("Patamon", date)
        # grammar induction
        grammar_induction_monkey("Omega", date)
        grammar_induction_monkey("Patamon", date)
        GrammarAlign("Monkey", date)
        # LoPS Complexity
        LoPSC_omplexity_monkey("Omega", date)
        LoPSC_omplexity_monkey("Patamon", date)

        # Performance
        reaction_time_monkey("Omega", date)
        reaction_time_monkey("Patamon", date)
        reward_monkey("Omega", date)
        reward_monkey("Patamon", date)


def generate_img_data():
    Fig2b()
    Fig2c()
    Fig3a()
    Fig3b()
    Fig4a2_4b2()
    Fig4a3_4b3()
    Fig4c2()
    Fig4c3()

    # Fig6
    static()
    toCsvMain()


def draw_img():
    plot_2b()
    plot_2c()
    plot_3a()
    plot_3b()
    plot_4a2_4b2()
    plot_4a3_4b3()
    plot_4c2()
    plot_4c3()
    plot_5()
    plot_6()

if __name__ == '__main__':
    generate_data()
    generate_img_data()
    draw_img()
