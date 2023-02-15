total_loss = 0
total_success = 0

for cl in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]:
    for win in [15]:
        success = 0
        loss = 0
        for i in range(4):
            path = '../../results_0201/results/cl_results_cl_'+str(cl)+'_w_'+str(win)+'_'+str(i)+'.txt'
            with open(path) as f:
                for s_line in f:
                    if s_line != 'nan'+'\n':
                        success += 1
                        loss += abs(float(s_line)-cl) #**2
        if success == 0:print('hog!')
        else: print(loss/success)
        total_loss += loss
        total_success += success  
print(total_loss/total_success)
print('---------------------')

total_loss = 0
total_success = 0

for cl in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]:
    path = '../../results_vae/results/cl_results_cl'+str(cl)+'.txt'
    success = 0
    loss = 0
    with open(path) as f:
        for s_line in f:
            if s_line != 'nan'+'\n':
                success += 1
                loss += abs(float(s_line)-cl) # **2
    print(loss/success)
    total_loss += loss
    total_success += success
print(total_loss/total_success)