from data_provider.data_loader import Dataset_Custom, Dataset_Custom2, Dataset_Pred, Dataset_TSF, Dataset_ETT_hour, Dataset_ETT_minute
from data_provider.lora_loader import Dataset_Lora
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'custom2': Dataset_Custom2,
    'lora': Dataset_Lora,
}


def data_provider(args, flag, drop_last_test=True, train_all=False):
    if flag=='lora':
        print('lora dataloader')
        Data = data_dict['lora']
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        percent = args.percent
        data_set = Data(
            root_path=args.root_path,
            lora_path=args.lora_path,
            technical_indicators=args.technical_indicators,
            size=[args.seq_len, args.label_len, args.pred_len],
            target=args.target,
            percent=percent,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader

    else:
        print('normal dataloader')
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1
        percent = args.percent
        max_len = args.max_len
        if flag == 'test':
            shuffle_flag = False
            drop_last = drop_last_test
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
            Data = Dataset_Pred
        elif flag == 'val':
            shuffle_flag = True
            drop_last = drop_last_test
            batch_size = args.batch_size
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            technical_indicators=args.technical_indicators,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            max_len=max_len,
            train_all=train_all,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
