from re import L
import sys
import pandas as pd
# python resana.py ./matrix_results/spmm_csr_gpu.csv res1020.csv
# scp zhr.eva7:/home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/res1020.csv /Users/zhang/Desktop
if __name__ == '__main__':
    infofile = sys.argv[1]
    outfile = sys.argv[2]
    libraries = int(sys.argv[3])
    info = pd.read_csv(infofile)
    lines = info.shape[0]
    trials = 25
    accu = 4 # round(4)
    if libraries == 3: 
        cnt = 0
        tmp = 0
        setloop = 0
        dataset = []
        taco = []
        taco_warp = []
        cusparse = []
        #nnZk = [997269504,9173289600,979378304,1811053824,5954876800,1489206272,2600480384,0,1000115712,0]
        #thpt_taco = []
        #thpt_taco_warp = []
        #thpt_cu = []
        j = 0
        for i in range(lines):
            tmp = tmp + info.loc[i].time
            if(cnt==24):
                setloop = setloop%3
                cnt = 0
                tmp = tmp/25
                #print("setloop %d"%(setloop))
                if(setloop == 0):
                    dataset.append(info.loc[i].tensor)
                if(setloop == 0):
                    taco.append(round(tmp,4))
                    j = len(taco) - 1
                    #thpt_taco.append(round(nnZk[j]/tmp/1e9,4))
                elif(setloop == 1):
                    taco_warp.append(round(tmp,4))
                    j = len(taco_warp) - 1
                    #thpt_taco_warp.append(round(nnZk[j]/tmp/1e9,4))
                else:
                    cusparse.append(round(tmp,4))
                    j = len(cusparse) - 1
                    #thpt_cu.append(round(nnZk[j]/tmp/1e9,4))
                tmp = 0

                setloop = setloop + 1

            else:
                cnt = cnt + 1 
        #outf = pd.DataFrame({'dataset':dataset,'taco':taco,'taco_warp':taco_warp,'cusparse':cusparse,'taco-thpt':thpt_taco,'wr-thpt':thpt_taco_warp,'taco-cu-thpt':thpt_cu})
        outf = pd.DataFrame({'dataset':dataset,'taco':taco,'taco_warp':taco_warp,'cusparse':cusparse})
        outf.to_csv(outfile,index=False,sep=',')
    
    elif libraries == 4:
        cnt = 0
        tmp = 0
        setloop = 0
        dataset = []
        pr = []
        taco_warp = []
        taco_row = []
        cusparse = []
        for i in range(lines):
            tmp = tmp + info.loc[i].time
            if(cnt==24):
                setloop = setloop%libraries
                cnt = 0
                tmp = tmp/25
                if(setloop == 0):
                    dataset.append(info.loc[i].tensor)
                if(setloop == 0):
                    pr.append(round(tmp,4))
                elif(setloop == 1):
                    taco_warp.append(round(tmp,4))
                elif(setloop == 2):
                    taco_row.append(round(tmp,4))
                else:
                    cusparse.append(round(tmp,4))
                tmp = 0

                setloop = setloop + 1

            else:
                cnt = cnt + 1 
        outf = pd.DataFrame({'dataset':dataset,'pr':pr,'taco-nnz':taco_warp,'taco-row':taco_row,'cusparse':cusparse})
        outf.to_csv(outfile,index=False,sep=',')

    elif libraries == 9:
        cnt = 0
        tmp = 0
        setloop = 0
        dfs = [[] for i in range(libraries)]
        dataset = []
        df_names = ['eb-pr-taco', 'eb-pr-dgSP', 'eb-sr-taco', 'eb-sr-dgSP', 'rb-pr-taco', 'rb-pr-dgSP', 'rb-sr-taco', 'rb-sr-dgSP', 'alg2']
        for i in range(lines):
            tmp = tmp + info.loc[i].time
            if(cnt==24):
                setloop = setloop%libraries
                cnt = 0
                tmp = tmp/25
                if(setloop == 0):
                    dataset.append(info.loc[i].tensor)
                dfs[setloop].append(round(tmp,4))
                tmp = 0
                setloop = setloop + 1
            else:
                cnt = cnt + 1

        df_dict = {'dataset':dataset}
        for i in range(libraries):
            df_dict[df_names[i]] = dfs[i]
        outf = pd.DataFrame(df_dict)
        outf.to_csv(outfile,index=False,sep=',')

    elif libraries == 5:
        cnt = 0
        tmp = 0
        tmps = []
        setloop = 0
        dfs = [[] for i in range(libraries)]
        dataset = []
        #df_names = ['eb-pr', 'eb-sr', 'rb-pr', 'rb-sr', 'alg2']
        df_names = ['eb-pr', 'rb-pr-8', 'rb-pr', 'rb-pr-4', 'alg2']
        for i in range(lines):
            tmps.append(info.loc[i].time)
            tmp = tmp + info.loc[i].time
            if(cnt==24):
                setloop = setloop%libraries
                cnt = 0
                tmp = tmp/25
                tmp = min(tmps)
                if(setloop == 0):
                    dataset.append(info.loc[i].tensor)
                dfs[setloop].append(round(tmp,4))
                tmp = 0
                tmps = []
                setloop = setloop + 1
            else:
                cnt = cnt + 1

        df_dict = {'dataset':dataset}
        for i in range(libraries):
            df_dict[df_names[i]] = dfs[i]
        outf = pd.DataFrame(df_dict)
        outf.to_csv(outfile,index=False,sep=',')
    elif libraries == 61:
        cnt = 0
        tmp = 0
        tmps = []
        setloop = 0
        dfs = [[] for i in range(libraries)]
        dataset = []
        #df_names = ['eb-pr', 'eb-sr', 'rb-pr', 'rb-sr', 'alg2']
        df_names = ['rbprtune0', 'rbprtune1', 'rbprtune2', 'rbprtune3', 'rbprtune4', 'rbprtune5', 'rbprtune6', 'rbprtune7', 'rbprtune8', 'rbprtune9', 'rbprtune10', 'rbprtune11', 'rbprtune12', 'rbprtune13', 'rbprtune14', 'rbprtune15', 'rbprtune16', 'rbprtune17', 'rbprtune18', 'rbprtune19', 'rbprtune20', 'rbprtune21', 'rbprtune22', 'rbprtune23', 'rbprtune24', 'rbprtune25', 'rbprtune26', 'rbprtune27', 'rbprtune28', 'rbprtune29', 'ebprtune0', 'ebprtune1', 'ebprtune2', 'ebprtune3', 'ebprtune4', 'ebprtune5', 'ebprtune6', 'ebprtune7', 'ebprtune8', 'ebprtune9', 'ebprtune10', 'ebprtune11', 'rbsrtune0', 'rbsrtune1', 'rbsrtune2', 'rbsrtune3', 'rbsrtune4', 'rbsrtune5', 'rbsrtune6', 'rbsrtune7', 'rbsrtune8', 'ebsrtune0', 'ebsrtune1', 'ebsrtune2', 'ebsrtune3', 'ebsrtune4', 'ebsrtune5', 'ebsrtune6', 'ebsrtune7', 'ebsrtune8','alg2']
        for i in range(lines):
            tmps.append(info.loc[i].time)
            tmp = tmp + info.loc[i].time
            if(cnt==24):
                setloop = setloop%libraries
                cnt = 0
                tmp = tmp/25
                tmp = min(tmps)
                if(setloop == 0):
                    dataset.append(info.loc[i].tensor)
                dfs[setloop].append(round(tmp,4))
                tmp = 0
                tmps = []
                setloop = setloop + 1
            else:
                cnt = cnt + 1

        df_dict = {'dataset':dataset}
        for i in range(libraries):
            df_dict[df_names[i]] = dfs[i]
        outf = pd.DataFrame(df_dict)
        outf.to_csv(outfile,index=False,sep=',')

    
