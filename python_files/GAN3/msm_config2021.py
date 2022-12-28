import numpy as np

AP = 1.4648587220804408 # exact
#AP = 1.5
REF_MIC = 63
DPATH = "td/"
CALIBFILE = "calib/Vogel_Typhoon_150321.xml"
tmptr = 17
c0 = np.sqrt(1.402*(8.3145*(273.15+tmptr)/0.02896))


FILES = [
    "2021-03-19_14-28-01_611275",
    "2021-03-19_14-28-36_994278",
    "2021-03-19_14-29-10_151158",
    "2021-03-19_14-29-25_410020",
    "2021-03-19_14-29-45_518152",
    "2021-03-19_14-30-36_346031",
    "2021-03-19_14-30-52_107924",
    "2021-03-19_14-31-16_250292",
    "2021-03-19_14-34-12_658217",
    "2021-03-19_14-34-26_902025",
    "2021-03-19_14-34-49_545295",
    "2021-03-19_14-35-08_728381",
    "2021-03-19_14-35-27_319079",
    "2021-03-19_14-35-47_230207",
    "2021-03-19_14-36-10_584524",
    "2021-03-19_14-36-37_496046",
    "2021-03-19_14-43-22_451338",
    "2021-03-19_14-43-40_659361",
    "2021-03-19_14-43-54_594153",
    "2021-03-19_14-44-08_842962",
    "2021-03-19_14-44-24_472851",
    "2021-03-19_14-44-38_596649",
    "2021-03-19_14-44-52_856457",
    "2021-03-19_14-45-07_563287",
    "2021-03-19_15-01-29_309657",
    "2021-03-19_15-04-05_343460",
    "2021-03-19_15-04-28_462768",
    "2021-03-19_15-04-48_202877",
    "2021-03-19_15-05-05_948884",
    "2021-03-19_15-05-22_079799",
    "2021-03-19_15-05-38_099701",
    "2021-03-19_15-06-15_183787", 
]

start = 0.07 # shift of the loudspeakers from the center (see doc/img/TZ_Ausleger_Vorne.pdf)
x_shift = -0.03 #  systematischer Fehler in der Ausrichtung der Focus Ebene!
y_shift = 0 #0.01
LOC = [
    (start+x_shift, 0.0+y_shift),  # 3 Uhr
    (start+.04+x_shift, 0.0+y_shift),
    (start+.08+x_shift, 0.0+y_shift),
    (start+.12+x_shift, 0.0+y_shift),
    (start+0.16+x_shift, 0.0+y_shift),
    (start+0.20+x_shift, 0.0+y_shift),
    (start+0.24+x_shift, 0.0+y_shift),
    (start+0.28+x_shift, 0.0+y_shift),
    (x_shift, -start+y_shift),  # 6 Uhr
    (x_shift, -start-.04+y_shift),
    (x_shift, -start-.08+y_shift),
    (x_shift, -start-.12+y_shift),
    (x_shift, -start-0.16+y_shift),
    (x_shift, -start-0.20+y_shift),
    (x_shift, -start-0.24+y_shift),
    (x_shift, -start-0.28+y_shift),
    (-start+x_shift, 0.0+y_shift),  # 9 Uhr
    (-start-.04+x_shift, 0.0+y_shift),
    (-start-.08+x_shift, 0.0+y_shift),
    (-start-.12+x_shift, 0.0+y_shift),
    (-start-0.16+x_shift, 0.0+y_shift),
    (-start-0.20+x_shift, 0.0+y_shift),
    (-start-0.24+x_shift, 0.0+y_shift),
    (-start-0.28+x_shift, 0.0+y_shift),
    (x_shift, start+y_shift),  # 12 Uhr
    (x_shift, start+.04+y_shift),
    (x_shift, start+.08+y_shift),
    (x_shift, start+.12+y_shift),
    (x_shift, start+0.16+y_shift),
    (x_shift, start+0.20+y_shift),
    (x_shift, start+0.24+y_shift),
    (x_shift, start+0.28+y_shift),
]
