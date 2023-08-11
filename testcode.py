import user_env_gym.controlutil as ctut

eng = ctut.start_matengine()

A, B, C, D, E = ctut.sopdt_dss()

print(ctut.lin_stab_td_SISO(eng, 100, 29.9, 99.7, A, B, C, D, E, intd = 2, outtd = 0))