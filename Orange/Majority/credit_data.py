# -*- coding: utf-8 -*-

import Orange
base = Orange.data.Table('Rule-based/Orange/credit_data/credit_db.csv')
base.domain

base_divider = Orange.evaluation.testing.sample(base, n=0.25)
base_trainning = base_divider[1]
base_tester = base_divider[0]

classifier = Orange.classification.MajorityLearner()
results = Orange.evaluation.testing.TestOnTestData(base_trainning,base_tester,[classifier])
print(Orange.evaluation.CA(results))

#baseline classifiers
from collections import Counter
print(Counter(str(d.get_class()) for d in base_tester))