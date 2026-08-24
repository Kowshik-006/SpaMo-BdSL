import torch,sys
c=torch.load(sys.argv[1],map_location='cuda')
print('epoch:',c.get('epoch'),'| global_step:',c.get('global_step'))
