for cl in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]:
    for win in [15]:
    # path = '../../results_04/results/cl_results_cl_'+str(cl)+'_g_0.0_w_13.txt'
    # path = '../../results_01/results/cl_results_cl_'+str(cl)+'_g_0.0_w_13.txt'
        success = 0
        num_all = 0
        for i in range(4):
            path = '../../results_0201/results/cl_results_cl_'+str(cl)+'_w_'+str(win)+'_'+str(i)+'.txt'
            with open(path) as f:
                for s_line in f:
                    num_all += 1
                    if s_line != 'nan'+'\n':
                        success += 1
        print(success/num_all)

# for i,cl in enumerate([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]):
#     path = '../../results_vae/results/cl_results_cl'+str(cl)+'.txt'
#     success = 0
#     num_all = 0
#     with open(path) as f:
#         for s_line in f:
#             num_all += 1
#             if s_line != 'nan'+'\n':
#                 success += 1
#     print(success/num_all)