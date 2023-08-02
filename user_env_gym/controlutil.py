import collections
import numbers
import numpy as np
import matlab.engine as mat
import matlab
import control as ct
import math

def start_matengine():
     eng = mat.start_matlab()
     return eng

def cartpole_dss(g, l, m_p, m_c, is_siso):
     tmp1 = (3 * g * m_p) / (4 - 3 * m_p)
     tmp2 = (3 * g * (m_p + m_c)) / (l * (4 - 3 * m_p))
     tmp3 = (4/3) / ((m_p + m_c) * (4/3 - m_p))
     tmp4 = 1 / (l * ((4/3) - m_p))

     A = [[0, 0, 1, 0],
          [0, 0, 0, 1],
          [0, tmp1, 0, 0],
          [0, tmp2, 0, 0]]
     B = [[0],
          [0],
          [tmp3],
          [tmp4]]
     if is_siso == True:
          C = [0, 1, 0, 0]
          D = [0]
     else:
          C = [[1, 0, 0, 0],
               [0, 1, 0, 0]]
          D = [[0],
               [0]]
     E = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
     
     return A, B, C, D, E

def lin_stability_SISO(eng, p, i, d, A, B, C, D, E, mode = 0):
     # p, i, d is pid coefficient
     # A, B, C, D, E is descriptive state-space form
     assert len(A) == len(A[0]) == len(B) == len(C), "Matrix size is not appropriate"

     # descriptive state space form of PID
     AP = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]
     BP = [[0],
          [0],
          [1]]
     CP = [d, p, i]
     DP = [0]
     EP = [[0, 1, 0],
          [0, 0, 1],
          [0, 0, 1]]

     sys = eng.dss(matlab.double(A),matlab.double(B),matlab.double(C),matlab.double(D), matlab.double(E))

     S = eng.struct('type', '.', 'subs', 'InputName')
     T = eng.struct('type', '.', 'subs', 'OutputName')
     sys = eng.subsasgn(sys, S, 'u')
     sys = eng.subsasgn(sys, T, 'y')

     pid = eng.dss(matlab.double(AP),matlab.double(BP),matlab.double(CP),matlab.double(DP), matlab.double(EP))
     pid = eng.subsasgn(pid, S, 'e')
     pid = eng.subsasgn(pid, T, 'u')

     sum = eng.sumblk('e = r - y')

     CLTF = eng.connect(pid, sys, sum, 'r', 'y')
     pole_list = eng.pole(CLTF)
     
     if mode == 0:
          max_pl_real = pole_list[0][0].real

          for pl in pole_list:
               if pl[0].real > max_pl_real:
                    max_pl_real = pl[0].real
          
          return max_pl_real
     elif mode == 1:
          criterion = 0
          for pl in pole_list:
               criterion += math.pow(math.e, pl[0].real)
          return criterion

def lin_stab_perf_SISO(eng, p, i, d, A, B, C, D, E, mode = 0):
     # p, i, d is pid coefficient
     # A, B, C, D, E is descriptive state-space form
     assert len(A) == len(A[0]) == len(B) == len(C), "Matrix size is not appropriate"

     # descriptive state space form of PID
     AP = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]
     BP = [[0],
          [0],
          [1]]
     CP = [d, p, i]
     DP = [0]
     EP = [[0, 1, 0],
          [0, 0, 1],
          [0, 0, 1]]

     sys = eng.dss(matlab.double(A),matlab.double(B),matlab.double(C),matlab.double(D), matlab.double(E))

     S = eng.struct('type', '.', 'subs', 'InputName')
     T = eng.struct('type', '.', 'subs', 'OutputName')
     sys = eng.subsasgn(sys, S, 'u')
     sys = eng.subsasgn(sys, T, 'y')

     pid = eng.dss(matlab.double(AP),matlab.double(BP),matlab.double(CP),matlab.double(DP), matlab.double(EP))
     pid = eng.subsasgn(pid, S, 'e')
     pid = eng.subsasgn(pid, T, 'u')

     sum = eng.sumblk('e = r - y')

     CLTF = eng.connect(pid, sys, sum, 'r', 'y')
     pole_list = eng.pole(CLTF)
     
     if mode == 0:
          max_pl_real = pole_list[0][0].real

          for pl in pole_list:
               if pl[0].real > max_pl_real:
                    max_pl_real = pl[0].real

          max_pl_imag = pole_list[0][0].imag

          for pl in pole_list:
               if pl[0].imag > max_pl_imag:
                    max_pl_imag = pl[0].imag
          
          return max_pl_real + 0.3 * max_pl_imag
     elif mode == 1:
          criterion = 0
          for pl in pole_list:
               criterion += math.pow(math.e, pl[0].real)

          for pl in pole_list:
               criterion += 0.3 * math.pow(math.e, pl[0].imag)

          return criterion

def lin_stability_MIMO(eng, p, i, d, A, B, C, D, E):
     # p, i, d are pid coefficient array (P[0] for position, P[1] for angle pid)
     # A, B, C, D, E is descriptive state-space form
     # current problem : cannot remove semi-zero pole which should not exist, so we're 'manually' excluding that pole.
     
     AP = [[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]]
     BP = [[0, 0],
           [0, 0],
           [1, 0],
           [0, 0],
           [0, 0],
           [0, 1]]
     CP = [d[0], p[0], i[0], d[1], p[1], i[1]]
     DP = [0, 0]
     EP = [[0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1]]

     sys = eng.dss(matlab.double(A),matlab.double(B),matlab.double(C),matlab.double(D), matlab.double(E))

     S = eng.struct('type', '.', 'subs', 'InputName')
     T = eng.struct('type', '.', 'subs', 'OutputName')
     sys = eng.subsasgn(sys, S, 'u')
     sys = eng.subsasgn(sys, T, 'y')

     pid1 = eng.pid(p[0], i[0], d[0])
     pid2 = eng.pid(p[1], i[1], d[1])
     pid = eng.dss(matlab.double(AP),matlab.double(BP),matlab.double(CP),matlab.double(DP), matlab.double(EP))
     pid = eng.subsasgn(pid, S, 'e')
     pid = eng.subsasgn(pid, T, 'u')

     sum = eng.sumblk('e = r - y', 2)

     CLTF = eng.connect(pid, sys, sum, 'r', 'y')
     pole_list = eng.pole(CLTF)
     
     max_pl_real = pole_list[0][0].real

     semizero = False

     for pl in pole_list:
          if semizero == False and abs(pl[0].real) < 1e-6:
               semizero = True
               continue
          if pl[0].real > max_pl_real:
               max_pl_real = pl[0].real

     return max_pl_real

def calISE(traj, refsig):
     # traj means trajectory, refsig means reference signal, delt means time interval

     assert isinstance(traj, collections.abc.Sequence), "Trajectory is not a sequence type."
     assert len(traj) >= 1, "Trajectory is empty array."
     assert (isinstance(refsig, numbers.Real) and isinstance(traj[0], numbers.Real)) or (isinstance(refsig, collections.abc.Sequence) and isinstance(traj[0], collections.abc.Sequence)) \
             or (isinstance(refsig, np.ndarray) and isinstance(traj[0], np.ndarray)), "refsig dimension doesn't match with trajectory"
     
     if isinstance(refsig, collections.abc.Sequence) and len(refsig) <= 1:
          print("ErrorMsg : length of refsig must be longer than 1 if it is sequence type.")
          return -1

     if isinstance(refsig, collections.abc.Sequence) and isinstance(traj[0], collections.abc.Sequence):
          if len(refsig) != len(traj[0]):
               print("ErrorMsg : length of refsig and traj[0] do not match.")
               return -1

     SISO = False
     ISE = [0]

     if isinstance(refsig, numbers.Real):
          SISO = True

     if SISO:
          for step in traj:
               ISE[0] += (refsig - step)**2
     else:
          ISE = [0] * len(refsig)
          for step in traj:
               for k in range(len(step)):
                    ISE[k] += (step[k] - refsig[k])**2
     
     return ISE

def calThreshold(trajlist):
     assert isinstance(trajlist, collections.abc.Sequence), "trajectory list should be a sequence"
     assert isinstance(trajlist[0], collections.abc.Sequence), "trajectory should be a sequence"

     SISO = False
     if isinstance(trajlist[0][0], numbers.Real):
          SISO = True

     thr = [-1e6]
     meanthr = [0]

     if SISO == False:
          thr = [-1e6] * len(trajlist[0][0])
          meanthr = [0] * len(trajlist[0][0])

     for traj in trajlist:
          for step in traj:
               if SISO:
                    if thr[0] < step:
                         thr[0] = step
               else:
                    for i in range(len(thr)):
                         if thr[i] < step[i]:
                              thr[i] = step[i]
          if SISO:
               meanthr[0] += thr[0]
               thr = [-1e6]
          else:
               for i in range(len(thr)):
                    meanthr[i] += thr[i]
               thr = [-1e6] * len(trajlist[0][0])

     for k in range(len(meanthr)):
          meanthr[k] = meanthr[k] / len(trajlist)
     if SISO:
          return meanthr[0]
     else:
          return meanthr