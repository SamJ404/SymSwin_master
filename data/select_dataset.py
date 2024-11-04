

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# modified by SAM J
#  based on KAIR-master (github: https://github.com/cszn/KAIR)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    # -----------------------------------------
    # super-resolution
    # -----------------------------------------
    if dataset_type in ['sr', 'super-resolution']:
        from data.dataset_sr import DatasetSR as D


    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
