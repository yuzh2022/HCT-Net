class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset=='isic2018':
            return r'F:\820\subject\dataset\MixSearch\ISIC2018\Task1_apply'
        elif dataset=='cvc':
            return r'/content/drive/MyDrive/Dataset/Mixsearch/CVC'
        elif dataset=='chaos':
            return r'F:\820\subject\dataset\MixSearch\CHAOS_CT'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
