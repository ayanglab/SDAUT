'''
# -----------------------------------------
Define Training Model
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

def define_Model(opt):
    model = opt['model']

    # --------------------------------------------------------
    # SDAUT
    # --------------------------------------------------------
    if model == 'sdaut_npi':
        from models.model_sdaut import MRI_SDAUT as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
