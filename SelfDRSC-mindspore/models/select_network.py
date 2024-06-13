# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    if net_type == 'srsc_rsg':
        from models.network_srsc_rsg import RSG as net
        netG = net(num_frames=opt_net['num_frames'],
                   n_feats=opt_net['n_feats'],
                   load_flow_net=opt_net['load_flow_net'],
                   flow_pretrain_fn=opt_net['flow_pretrain_fn'])

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    return netG
