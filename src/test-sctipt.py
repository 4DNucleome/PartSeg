from __future__ import print_function

from partseg.backend import StatisticProfile

aa = StatisticProfile("aa", [("Moment of inertia", "Moment of inertia")], False, None)

print(aa.parse_statistic("Mass/(Volume/Moment of inertia)"))
print(aa.parse_statistic("(Mass aa/Moment of inertia[thr=2200]/(Volume/Moment of inertia)"))
print(aa.parse_statistic("Mass"))
