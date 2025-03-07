class SB_App:
    # Initialize the class    
    def __init__(self,AppID='',BAY=None,StagingBinsLocs={}):
        self.AppID = AppID
        self.BAY = BAY          # BAY is BAY object such as BAY2Obj 
        self.StagingBinsLocs = StagingBinsLocs
        self.StagingBins = {}  # sorted staging bins for each location (key) od warehouse
                               # such as {'(91, 17))':['3A', '1A', '4A', '5A', '2A', '6A'], ...}

    def __repr__(self):
        return ('({0})'.format(self.AppID))      

    ##########################
    # Staging Bins application
    
    # sorted distances between a_loc and those staging bins in self.StagingBinLocs
    def Dist_StagingBins(self,a_loc): 
        ret = {}
        for k,v in self.StagingBinsLocs.items():
            #print('k = ',k, '   v = ', v)
            ret[k] = self.BAY.cur_DistMatrix[a_loc].loc[v]
            
            ret = dict(sorted(ret.items(), key=lambda item: item[1]))
            ret_list = list(ret.keys())
        return (ret,ret_list)
        
    # Sorted distances between all locations in itemLocs and those staging bins in StagingBinLocs
    def Dist_StagingBins2(self,itemLocs=[],StagingBinLocs={}):
        # if StagingBinLocs is not empty, update self.StagingBinsLocs
        if bool(StagingBinLocs):
            self.StagingBinsLocs = StagingBinLocs.copy()
        
        self.BAY.cur_DistMatrix = self.BAY.FullDistMatrix.copy()  
        
        ret = {}    
        for i in itemLocs:
            #print('i = ',i)
            ret[i] = self.Dist_StagingBins(str(i))
                
        return ret

    # create full self.StagingBins: Sorted distances between all locations in self.BAY.FullDistMatrix 
    #                               and those staging bins in self.StagingBinLocs such as 
    #   {'(51, 3)': ({'2A': 108.05277777777775, '4A': 109.32347670250894, '3A': 191.017876344086, 
    #                 '6A': 191.34444444444443, '1A': 191.6583333333333, '5A': 274.95},
    #                 ['2A', '4A', '3A', '6A', '1A', '5A']), ......}
    def Dist_StagingBins3(self,StagingBinLocs={}):
        # if StagingBinLocs is not empty, update self.StagingBinsLocs
        if bool(StagingBinLocs):
            self.StagingBinsLocs = StagingBinLocs.copy()
         
        self.BAY.cur_DistMatrix = self.BAY.FullDistMatrix.copy()    
        
        ret = {}    
        for i in self.BAY.FullDistMatrix.columns:
            #print('i = ',i)
            ret[i] = self.Dist_StagingBins(str(i))
        
        self.StagingBins = ret
        
        return ret

    ########################################
    # For a given agent, find an optimal path: agent -> itemLoc -> StagingBin Loc
    # here, agent = (agentID,agentLoc) such as ('agent1','(91, 17)')
    #       AvailableStagingBins is, for example, ['1A','3A','4A']
    def OptimalPath_StagingBins(self,itemLocs=[],agent=('',''),AvailableStagingBins=[]):
        min_d = 999999.0
        itemLoc = ''
        StagingB = ''
        for i in itemLocs:
            #print('******** agent = ', agent)
            #print('i = ', i)
            #print('self.StagingBins[i][0][self.StagingBins[i][1][0]] = ',self.StagingBins[i][0][self.StagingBins[i][1][0]])
            d1 = self.BAY.cur_DistMatrix[self.BAY.eqivalent_loc(agent[1])].loc[self.BAY.eqivalent_loc(i)]
            #d2 = self.StagingBins[i][0][self.StagingBins[i][1][0]]
            
            for a_ID in self.StagingBins[i][1]:
                #print('a_ID in 000 = ', a_ID)
                if a_ID in AvailableStagingBins:
                    d2 = self.StagingBins[i][0][a_ID]
                    break
                                
            #print('d1 = ', d1, '    d2 = ', d2)
            if min_d > d1+d2:
                min_d = d1+d2
                itemLoc = i
                StagingB = a_ID
                    
        return (agent,itemLoc,StagingB,min_d)


    # out of given agents, find the best agent with optimal path: agent -> itemLoc -> StagingBin Loc
    # here, agents = [(agentID1,agentLoc1),(agentID2,agentLoc2), ...] such as
    #       [('agent1','(91, 17)'),('agent12,'(86, 34)'), ...]
    def OptimalPath_StagingBins2_1(self,itemLocs=[],agents=[],AvailableSBins=[]):
        assert bool(self.StagingBins)
        assert bool(self.StagingBinsLocs)
        
        if len(agents) == 0: return {}
        
        self.BAY.cur_DistMatrix = self.BAY.FullDistMatrix.copy()           
        # main loop
        itemLocs_c = itemLocs.copy()
        AvailableSBins_c = AvailableSBins.copy()
        
        min_a = agents[0]
        
        ret = self.OptimalPath_StagingBins(itemLocs=itemLocs_c,agent=min_a,AvailableStagingBins=AvailableSBins_c)
        #print('ret in 111 = ', ret)
        min_ret = ret  # ret: (agent,itemLoc,StagingB,min_d)
#        itemLocs_c.remove(min_ret[1])
#        AvailableSBins_c.remove(min_ret[2])
        
        for a in agents[1:]:
            #print('agent = ',a)
            #print('itemLocs_c = ',itemLocs_c)
            #print('AvailableSBins_c = ',AvailableSBins_c)
            # here, ret = (agent,itemLoc,StagingB) where agent = (agentID,agentLoc)
            ret = self.OptimalPath_StagingBins(itemLocs=itemLocs_c,agent=a,AvailableStagingBins=AvailableSBins_c)
            #print('ret in 222 = ', ret)
            
            if min_ret[3] > ret[3]:
                min_ret = ret
                
#            itemLocs_c.remove(ret[1])
#            AvailableSBins_c.remove(ret[2])
            
        return {'task':min_ret}
    
    # Find optimal paths for all agents: agent -> itemLoc -> StagingBin Loc
    # here, agents = [(agentID1,agentLoc1),(agentID2,agentLoc2), ...] such as
    #       [('agent1','(91, 17)'),('agent12,'(86, 34)'), ...]
    def OptimalPath_StagingBins2_2(self,itemLocs=[],agents=[],AvailableSBins=[]):
        assert bool(self.StagingBins)
        assert bool(self.StagingBinsLocs)
        
        self.BAY.cur_DistMatrix = self.BAY.FullDistMatrix.copy()           
        
        # main loop
        itemLocs_c = itemLocs.copy()
        agents_c = agents.copy()
        
        if len(AvailableSBins) == 0:
            AvailableSBins = list(self.StagingBinsLocs.keys())
        else:
            AvailableSBins = AvailableSBins.copy()
            
        n_tasks = min(len(itemLocs),len(agents),len(AvailableSBins))
            
        tasks = {}
        taskID = 0
        while True:
            if taskID == n_tasks: break
            #print('*** agents_c = ',agents_c)
            #print('*** itemLocs_c = ',itemLocs_c)
            #print('*** AvailableSBins = ',AvailableSBins)
            # here, ret = (agent,itemLoc,StagingB) where agent = (agentID,agentLoc)
            #ret = self.OptimalPath_StagingBins(itemLocs=itemLocs_c,agent=agents[taskID],AvailableStagingBins=AvailableSBins)
            ret = self.OptimalPath_StagingBins2_1(itemLocs=itemLocs_c,agents=agents_c,AvailableSBins=AvailableSBins)
            #print('*** ret = ', ret)    # ret: {'task':(agent,itemLoc,StagingB,min_d)}
                       
            taskID = taskID + 1
            tasks['Task'+str(taskID)] = ret['task']
            itemLocs_c.remove(ret['task'][1])
            AvailableSBins.remove(ret['task'][2])
            agents_c.remove(ret['task'][0])
            
        return tasks 
    
    # By calling OptimalPath_StagingBins2_2(...), Find optimal paths for all agents: 
    # agent -> itemLoc -> StagingBin Loc untill no any staging bin is available.
    # here, (1) agents = [(agentID1,agentLoc1),(agentID2,agentLoc2), ...] such as
    #              [('agent1','(91, 17)'),('agent12,'(86, 34)'), ...]
    #       (2) AvailableSBins is list of staging bin IDs such as ['1A','2A',...,'6A']
    #       (3) itemLocs2 is list of item locations such as  ['(86, 7)','(56, 4)','(56, 41)','(69, 23)','(68, 35)']
    # Note: in OptimalPath_StagingBins2_3, each agent may takes care of multiple items while,
    #       in OptimalPath_StagingBins2_2, each agent only takes care of one item.
    def OptimalPath_StagingBins2_3(self,itemLocs=[],agents=[],AvailableSBins=[]):
        tasks = {}
        itemLocs_c = itemLocs.copy()
        agents_c = agents.copy()

        if len(AvailableSBins) == 0:
            AvailableSBins_c = list(self.StagingBinsLocs.keys())
        else:
            AvailableSBins_c = AvailableSBins.copy()
        
        #print('AvailableSBins_c = ',AvailableSBins_c)
        #print('itemLocs_c = ',itemLocs_c)
        
        # main loop
        while True:
            if len(AvailableSBins_c) == 0 or len(itemLocs_c) == 0: break
            ret_tasks = self.OptimalPath_StagingBins2_2(itemLocs=itemLocs_c,agents=agents_c,AvailableSBins=AvailableSBins_c)
            
            agents_c2=[]
            for task in ret_tasks.keys():
                print('ret_tasks[task] = ',ret_tasks[task]) # for example, ret_tasks[task] = (('agent1','(51,17)'),'(55,17)','2A',49.3848)
                AvailableSBins_c.remove(ret_tasks[task][2])
                itemLocs_c.remove(ret_tasks[task][1])
                
                # updating agents_c
                for agent in agents_c:
                    if agent[0] == ret_tasks[task][0][0]: # for example, ret_tasks[task] = (('agent1','(51,17)'),'(55,17)','2A',49.3848)
                        #print('agent = ', agent)
                        agent = (agent[0],self.StagingBinsLocs[ret_tasks[task][2]])  # update agent's location
                        
                        agents_c2.append(agent)
                        break
                
            agents_c = agents_c2.copy()
            
            # updating tasks
            taskID = len(tasks)
            for k in ret_tasks.keys():
                tasks['task'+str(int(k[4:])+taskID)] = ret_tasks[k]
        
        return (tasks,itemLocs_c) # tasks: all tasks (with all staging bins); itemLocs_c: the items which have not been handled
    

    # Based on OptimalPath_StagingBins2_3(...), balancing the numbers of items assigned 
    # to each Staging Bin to avoid Staging Bins' bottle neck.
    def OptimalPath_StagingBins2_4(self,itemLocs=[],agents=[],AvailableSBins=[]):
        itemLocs_c = itemLocs.copy()
        agents_c = agents.copy()

        if len(AvailableSBins) == 0:
            AvailableSBins_c = list(self.StagingBinsLocs.keys())
        else:
            AvailableSBins_c = AvailableSBins.copy()

        ret = self.OptimalPath_StagingBins2_3(itemLocs=itemLocs_c,agents=agents_c,AvailableSBins=AvailableSBins_c)
        
        tasks = ret[0].copy()
        remain_items = ret[1].copy()
        
        # main loop
        while len(remain_items) > 0:           
            # update agents' locations
            agentsDict = dict(agents_c)
            for agent_k in agentsDict.keys():    # for example, agentsDict[agent_k] = '(51,17)'
                for task in ret[0].keys():       # for example, tasks[task] = (('agent1','(51,17)'),'(55,17)','2A',49.3848)
                    if agent_k == tasks[task][0][0]:   
                        agentsDict[agent_k] = self.StagingBinsLocs[tasks[task][2]]
                
            agents_c = [] 
            for agent_k in agentsDict.keys():
                agents_c.append((agent_k,agentsDict[agent_k]))
                                
            i = len(ret[0]) # current task number
            ret = self.OptimalPath_StagingBins2_3(itemLocs=remain_items,agents=agents_c,AvailableSBins=AvailableSBins_c)
            remain_items = ret[1].copy()
            
            for k in ret[0].keys():
                print('i = ',i)
                i = i + 1
                tasks['task' + str(i)] = ret[0][k]
        
        return tasks

    
    ################################################################################################
    # NOTE: OptimalPath_StagingBins2_* is suggested rather than following OptimalPath_StagingBins3_*
    # For all agents, find optimal paths: agent -> itemLoc -> StagingBin Loc for each agent.
    # here, agents = [(agentID1,agentLoc1),(agentID2,agentLoc2), ...] such as
    #       [('agent1','(91, 17)'),('agent12,'(86, 34)'), ...]
    def OptimalPath_StagingBins3_1(self,itemLocs=[],agents=[]):
        assert bool(self.StagingBins)
        assert bool(self.StagingBinsLocs)
        
        self.BAY.cur_DistMatrix = self.BAY.FullDistMatrix.copy()           
        n_tasks = min(len(itemLocs),len(agents),len(self.StagingBinsLocs))
        
        # main loop
        itemLocs_c = itemLocs.copy()
        AvailableSBins = list(self.StagingBinsLocs.keys())
        tasks = {}
        taskID = 0
        while True:
            if taskID == n_tasks: break
            print('*** agent = ',agents[taskID])
            print('*** itemLocs_c = ',itemLocs_c)
            print('*** AvailableSBins = ',AvailableSBins)
            # here, ret = (agent,itemLoc,StagingB) where agent = (agentID,agentLoc)
            ret = self.OptimalPath_StagingBins(itemLocs=itemLocs_c,agent=agents[taskID],AvailableStagingBins=AvailableSBins)
            print('*** ret = ', ret)
                       
            taskID = taskID + 1
            tasks['Task'+str(taskID)] = ret
            itemLocs_c.remove(ret[1])
            AvailableSBins.remove(ret[2])
            
        return tasks    

    # for given ret_tasks retuned by OptimalPath_StagingBins2_Original_Original(...),
    # swop staging bins across different tasks to further improve the sulution ret_tasks
    def OptimalPath_StagingBins3_2(self,tasks={}):
        print('tasks = ', tasks)
        
        ret_tasks = tasks.copy()
        while True:
            is_swop = False
            tasks_c = ret_tasks.copy()
            tasks_c2 = ret_tasks.copy()
            
            for k1 in tasks_c.keys():
                print('****** task1 = ', tasks_c[k1])
                #print('d = ',self.len_task(tasks_c[k1]))
                if is_swop: break # exit k1-loop
                
                task1 = tasks_c[k1]
                del tasks_c2[k1]
                for k2 in tasks_c2.keys():
                    print('****** task2 = ', tasks_c[k2])
                    task2 = tasks_c[k2]
                    
                    ret, is_swop = self.swopper(task1, task2)
                    print('ret = ', ret)
                    print('is_swop = ',is_swop)
                    
                    ret_tasks[k1] = ret[0]
                    ret_tasks[k2] = ret[1]
                    
                    if is_swop: break # exit k2-loop
             
            if not is_swop: break # exit while-loop 
        
        return ret_tasks
    

    # return length of a given task: len(agent -> item -> Staging Bin)  
    # here, task is, for example, ('agent1','(91, 17)')
    def len_task(self,task):
        d1 = self.BAY.cur_DistMatrix[task[0][1]].loc[task[1]]
        d2 = self.BAY.cur_DistMatrix[task[1]].loc[self.StagingBinsLocs[task[2]]]
        
        return d1 + d2
    
    # For fiven two tasks, return best solution by swapping their staging bins if necessary.
    # here, task1 and task2 are of data format
    #       (('agent2', '(91, 34)'), '(68, 34)', '6A')
    def swopper(self,task1,task2):
        d1 = self.len_task(task1)
        d2 = self.len_task(task2)
        
        task1_c = (task1[0],task1[1],task2[2])
        task2_c = (task2[0],task2[1],task1[2])
        
        d1_c = self.len_task(task1_c)
        d2_c = self.len_task(task2_c) 
        
        if d1 + d2 <= d1_c + d2_c:
            return ((task1,task2),False)
        else:
            return ((task1_c,task2_c),True)