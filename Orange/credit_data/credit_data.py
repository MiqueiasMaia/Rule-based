# -*- coding: utf-8 -*-

import Orange
base = Orange.data.Table('Rule-based/Orange/credit_data/credit_db.csv')
base.domain

base_divider = Orange.evaluation.testing.sample(base, n=0.25)
base_trainning = base_divider[1]
base_tester = base_divider[0]

cn2_learner = Orange.classification.rules.CN2Learner()
classifier = cn2_learner(base_trainning)

for rules in classifier.rule_list:
    print(rules)

result = Orange.evaluation.testing.TestOnTestData(base_trainning, base_tester, [classifier])
print(Orange.evaluation.CA(result))

# classifier = Orange.classification.MajorityLearner()
# result = Orange.evaluation.testing.TestOnTestData(base_trainning, base_tester, [classifier])
# print(Orange.evaluation.CA(result))