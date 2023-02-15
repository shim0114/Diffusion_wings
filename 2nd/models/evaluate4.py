import numpy as np

coords = np.load("../wing_data/standardized_NandJ_coords.npz")

max1 = 0
for cl in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]:
    wing_lst = []
    for i in range(4):
        path = '../../results_0201/results/cl_results_cl_'+str(cl)+'_w_15_'+str(i)+'.txt'
        pather = '../../results_0201/tmp/'
        with open(path) as f:
            idx = 0
            for s_line in f:
                if s_line != 'nan'+'\n':
                    wing_tmp = np.zeros(496)
                    with open(pather+'cdiffusion_cl_'+str(cl)+'_g_0.0_w_15_'+str(i*25+idx)+'.txt') as g:
                        for j ,wing_line in enumerate(g):
                            wing_tmp[j], wing_tmp[248+j] = wing_line.split()
                    wing_lst.append(wing_tmp)
                idx += 1
    max2 = 0
    for wing_gen in wing_lst:
        max2 = max(max2, np.min(\
        np.sum(\
        np.sqrt(\
        ((coords['arr_0'] - wing_gen)**2)[:,:248]+\
        ((coords['arr_0'] - wing_gen)**2)[:,248:]),axis=1))/248)
    max1 = max(max1,max2)

print(max1)



max1 = 0
for i,cl in enumerate([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]):
    path = '../../results_vae/results/cl_results_cl'+str(cl)+'.txt'
    success = 0
    wing_lst = []
    loss = 0
    with open(path) as f:
        idx = 0
        for s_line in f:
            if s_line != 'nan'+'\n':
                success += 1
                wing_tmp = np.zeros(496)
                with open('../../results_vae/tmp/cdiffusion_cl_'+str(cl)+'_'+str(idx)+'.txt') as g:
                    for j ,wing_line in enumerate(g):
                        wing_tmp[j], wing_tmp[248+j] = wing_line.split()
                wing_lst.append(wing_tmp)
            idx += 1
    max2 = 0
    for wing_gen in wing_lst:
        max2 = max(max2, np.min(\
        np.sum(\
        np.sqrt(\
        ((coords['arr_0'] - wing_gen)**2)[:,:248]+\
        ((coords['arr_0'] - wing_gen)**2)[:,248:]),axis=1))/248)
    max1 = max(max1,max2)
print(max1)
