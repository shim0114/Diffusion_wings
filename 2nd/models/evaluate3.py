import numpy as np
total_wing_lst = []
total_success = 0


for i,cl in enumerate([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]):
    success=0
    for i in range(4):
        path = '../../results_0201/results/cl_results_cl_'+str(cl)+'_w_15_'+str(i)+'.txt'
        pather = '../../results_0201/tmp/'
        with open(path) as f:
            idx = 25*i
            for s_line in f:
                if s_line != 'nan'+'\n':
                    success+=1
                    wing_tmp = np.zeros(496)
                    with open(pather+'cdiffusion_cl_'+str(cl)+'_g_0.0_w_15_'+str(idx)+'.txt') as g:
                        for j ,wing_line in enumerate(g):
                            wing_tmp[j], wing_tmp[248+j] = wing_line.split()
                    total_wing_lst.append(wing_tmp)
                idx += 1
    total_success+=success

print(np.sum(\
    np.sqrt(\
    ((total_wing_lst - np.mean(total_wing_lst, axis=0))**2)[:,:248] + \
    ((total_wing_lst - np.mean(total_wing_lst, axis=0))**2)[:,248:]))
    / (total_success*248))

total_wing_lst = []
total_success = 0
for i,cl in enumerate([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]):
    path = '../../results_vae/results/cl_results_cl'+str(cl)+'.txt'
    success = 0
    wing_lst = []
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
                total_wing_lst.append(wing_tmp)
            idx += 1
    total_success += success

print(np.sum(\
    np.sqrt(\
    ((total_wing_lst - np.mean(total_wing_lst, axis=0))**2)[:,:248] + \
    ((total_wing_lst - np.mean(total_wing_lst, axis=0))**2)[:,248:]))
    / (total_success*248))