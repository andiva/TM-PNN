import numpy as np
import matplotlib.pyplot as plt

from ocelot import *
from ocelot.cpbd.optics import *
from ocelot.cpbd.moga import *
from ocelot.gui.accelerator import *


bpm_01 = Monitor()
qf1 = Quadrupole(l= 0.281865276811,k1= 2.90604450103)
sh1a = Sextupole(l= 0.0955475514614,k2= 0)
qd2 = Quadrupole(l= 0.202560809098,k1= -3.12460925425)
dl1a_1 = SBend(l= 0.341678557602,angle= 0.0060061926,k1= 0,e1= 0.0079280963,e2= -0.0019219037)
dl1a_2 = SBend(l= 0.341678220925,angle= 0.003525,k1= 0,e1= 0.0019219037,e2= 0.0016030963)
dl1a_3 = SBend(l= 0.341678145898,angle= 0.002675,k1= 0,e1= -0.0016030963,e2= 0.0042780963)
dl1a_4 = SBend(l= 0.341678106809,angle= 0.0021,k1= 0,e1= -0.0042780963,e2= 0.0063780963)
dl1a_5 = SBend(l= 0.341678078229,angle= 0.00155,k1= 0,e1= -0.0063780963,e2= 0.0079280963)
qd3 = Quadrupole(l= 0.154787033367,k1= -2.91952200561)
sd1a = Sextupole(l= 0.158608935426,k2= -364.776406088)
bpm_02 = Monitor()
qf4a = Quadrupole(l= 0.202560809098,k1= 2.85194577762)
dispbumpcenter = Marker()
sf2ah = Sextupole(l= 0.0955475514614,k2= 358.280040913)
sf2amarker = Marker()
qf4b = Quadrupole(l= 0.202560809098,k1= 2.85194577762)
of1b = Octupole(l= 0.0859927963153,k3= -51832.8988549)
bpm_03 = Monitor()
sd1b = Sextupole(l= 0.158608935426,k2= -343.364031051)
qd5 = Quadrupole(l= 0.202560809098,k1= -3.15657557959)
dl2b_1 = SBend(l= 0.341678078229,angle= 0.00155,k1= 0,e1= 0.0073530963,e2= -0.0058030963)
dl2b_2 = SBend(l= 0.341678106809,angle= 0.0021,k1= 0,e1= 0.0058030963,e2= -0.0037030963)
dl2b_3 = SBend(l= 0.341678145898,angle= 0.002675,k1= 0,e1= 0.0037030963,e2= -0.0010280963)
dl2b_4 = SBend(l= 0.341678220925,angle= 0.003525,k1= 0,e1= 0.0010280963,e2= 0.0024969037)
dl2b_5 = SBend(l= 0.341678379762,angle= 0.0048561926,k1= 0,e1= -0.0024969037,e2= 0.0073530963)
bpm_04 = Monitor()
qf6 = Quadrupole(l= 0.37072449967,k1= 4.94993244327)
dq1 = SBend(l= 0.982263725187,angle= 0.0146,k1= -2.05256231734,e1= 0.0073,e2= 0.0073)
qf8 = Quadrupole(l= 0.462450149073,k1= 4.92090678216)
sh2b = Sextupole(l= 0.0955475514614,k2= 0)
bpm_05 = Monitor()
dq2c_1 = SBend(l= 0.382191187119,angle= 0.003925,k1= -1.62895801857,e1= 0.003925,e2= 0)
cellcenter = Marker()
dq2c_2 = SBend(l= 0.382191187119,angle= 0.003925,k1= -1.62895801857,e1= 0,e2= 0.003925)
bpm_06 = Monitor()
bpm_07 = Monitor()
dl2d_1 = SBend(l= 0.341678379762,angle= 0.0048561926,k1= 0,e1= 0.0073530963,e2= -0.0024969037)
dl2d_2 = SBend(l= 0.341678220925,angle= 0.003525,k1= 0,e1= 0.0024969037,e2= 0.0010280963)
dl2d_3 = SBend(l= 0.341678145898,angle= 0.002675,k1= 0,e1= -0.0010280963,e2= 0.0037030963)
dl2d_4 = SBend(l= 0.341678106809,angle= 0.0021,k1= 0,e1= -0.0037030963,e2= 0.0058030963)
dl2d_5 = SBend(l= 0.341678078229,angle= 0.00155,k1= 0,e1= -0.0058030963,e2= 0.0073530963)
sd1d = Sextupole(l= 0.158608935426,k2= -343.364031051)
bpm_08 = Monitor()
of1d = Octupole(l= 0.0859927963153,k3= -51832.8988549)
sf2eh = Sextupole(l= 0.0955475514614,k2= 358.280040913)
sf2emarker = Marker()
bpm_09 = Monitor()
sd1e = Sextupole(l= 0.158608935426,k2= -364.776406088)
dl1e_1 = SBend(l= 0.341678078229,angle= 0.00155,k1= 0,e1= 0.0079280963,e2= -0.0063780963)
dl1e_2 = SBend(l= 0.341678106809,angle= 0.0021,k1= 0,e1= 0.0063780963,e2= -0.0042780963)
dl1e_3 = SBend(l= 0.341678145898,angle= 0.002675,k1= 0,e1= 0.0042780963,e2= -0.0016030963)
dl1e_4 = SBend(l= 0.341678220925,angle= 0.003525,k1= 0,e1= 0.0016030963,e2= 0.0019219037)
dl1e_5 = SBend(l= 0.341678557602,angle= 0.0060061926,k1= 0,e1= -0.0019219037,e2= 0.0079280963)
sh3e = Sextupole(l= 0.0955475514614,k2= 0)

bpm_10 = Monitor()
idmarker = Marker()


def add(lattice, element, at):
    at -= element.l/2
    length = sum([el.l for el in lattice])
    # print(f'at position: length={length}, at={at}')
    if at<=length:
        print(f'wrong at position: length={length}, at={at}, at is shifted. {element}')
        at = length
    drift_length = at - length
    if drift_length!=0:
        lattice.append(Drift(l=drift_length))
    lattice.append(element)
    return


def get_lattice():
    lattice = []
    add(lattice, sf2amarker, at = 1.5) # inserted
    add(lattice, bpm_01, at = 2.53334777945)
    add(lattice, qf1, at = 2.7230096691)
    add(lattice, sh1a, at = 3.01681838984)
    add(lattice, qd2, at = 3.38563193848)
    add(lattice, dl1a_1, at = 3.70265897102)
    add(lattice, dl1a_2, at = 4.04433736028)
    add(lattice, sf2amarker, at = 4.3) # inserted
    add(lattice, dl1a_3, at = 4.38601554369)
    add(lattice, dl1a_4, at = 4.72769367005)
    #add(lattice, sf2amarker, at = 5) # inserted
    add(lattice, dl1a_5, at = 5.06937176257)
    add(lattice, qd3, at = 5.36251166755)
    add(lattice, sd1a, at = 5.59087031555)
    add(lattice, bpm_02, at = 6.19076889907)
    add(lattice, qf4a, at = 6.34555593244)
    #add(lattice, dispbumpcenter, at = 6.51849700059)
    add(lattice, sf2ah, at = 6.56627077632)
    add(lattice, sf2amarker, at = 6.61404455205)
    add(lattice, sf2ah, at = 6.66181832778)
    add(lattice, qf4b, at = 6.88253317165)
    add(lattice, of1b, at = 7.08413850638)
    add(lattice, bpm_03, at = 7.18064153336)
    add(lattice, sd1b, at = 7.63721878654)
    add(lattice, qd5, at = 7.8894643224)
    add(lattice, dl2b_1, at = 8.20649111525)
    add(lattice, sf2amarker, at = 8.5) # inserted
    add(lattice, dl2b_2, at = 8.54816920777)
    add(lattice, dl2b_3, at = 8.88984733412)
    add(lattice, dl2b_4, at = 9.23152551753)
    add(lattice, dl2b_5, at = 9.57320381788)
    add(lattice, bpm_04, at = 9.83385770613)
    add(lattice, qf6, at = 10.0727265848)
    add(lattice, dq1, at = 10.8352134935)
    add(lattice, sf2amarker, at = 11.2) # inserted
    add(lattice, qf8, at = 11.643563227)
    add(lattice, sh2b, at = 12.0811710127)
    add(lattice, bpm_05, at = 12.1642973824)
    add(lattice, dq2c_1, at = 12.4088996048)
    #add(lattice, cellcenter, at = 12.5999951984)
    add(lattice, dq2c_2, at = 12.7910907919)
    add(lattice, bpm_06, at = 13.0356930143)
    add(lattice, qf8, at = 13.5564271698)
    add(lattice, sf2amarker, at = 14) # inserted
    add(lattice, dq1, at = 14.3647769032)
    add(lattice, qf6, at = 15.127263812)
    add(lattice, bpm_07, at = 15.3661326906)
    add(lattice, dl2d_1, at = 15.6267865789)
    add(lattice, dl2d_2, at = 15.9684648792)
    add(lattice, dl2d_3, at = 16.3101430626)
    add(lattice, dl2d_4, at = 16.651821189)
    add(lattice, sf2amarker, at = 16.7) # inserted
    add(lattice, dl2d_5, at = 16.9934992815)
    add(lattice, qd5, at = 17.3105260744)
    add(lattice, sd1d, at = 17.5627716102)
    add(lattice, bpm_08, at = 18.0193488634)
    add(lattice, of1d, at = 18.1158518904)
    add(lattice, qf4b, at = 18.317457224)
    #add(lattice, dispbumpcenter, at = 18.4903982921)
    add(lattice, sf2eh, at = 18.5381720678)
    add(lattice, sf2emarker, at = 18.5859458436)
    add(lattice, sf2eh, at = 18.6337196193)
    add(lattice, qf4a, at = 18.8544344632)
    add(lattice, bpm_09, at = 19.0092214965)
    add(lattice, sd1e, at = 19.6091200801)
    add(lattice, qd3, at = 19.8374787281)
    add(lattice, dl1e_1, at = 20.130618633)
    add(lattice, dl1e_2, at = 20.4722967256)
    add(lattice, dl1e_3, at = 20.8139748519)
    add(lattice, sf2amarker, at = 21.0) # inserted
    add(lattice, dl1e_4, at = 21.1556530353)
    add(lattice, dl1e_5, at = 21.4973314246)
    add(lattice, qd2, at = 21.8143584571)
    add(lattice, sh3e, at = 22.1831720058)
    add(lattice, qf1, at = 22.4769807265)
    add(lattice, bpm_10, at = 22.6666426162)
    add(lattice, sf2amarker, at = 23.8) # inserted
    # add(lattice, idmarker, at = 25.1999903956)
    add(lattice, bpm_10, at = 25.1999903956)

    print(f'length: {sum([el.l for el in lattice])}')
    return lattice


def get_transfermaps(dim = 2):
    sequence = get_lattice()
    method = MethodTM()
    method.global_method = SecondTM

    lattice = MagneticLattice(sequence,  method=method)

    for i, tm in enumerate(get_map(lattice, lattice.totalLen, Navigator(lattice))):
        R = tm.r_z_no_tilt(tm.length, 0)[:dim, :dim]
        T = tm.t_mat_z_e(tm.length, 0)[:dim, :dim, :dim].reshape((dim, -1))
        yield R, T, type(lattice.sequence[i]).__name__, lattice.sequence[i].l


def main():
    sequence = get_lattice()
    method = MethodTM()
    method.global_method = SecondTM

    lattice = MagneticLattice(sequence,  method=method)

    tws = twiss(lattice)
    plot_opt_func(lattice,tws)
    plt.show()

    return


if __name__ == "__main__":
    main()