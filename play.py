import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset import MDataset, createDataCSV

from src.model import LightXML

if __name__ == '__main__':
    dataset= "Eurlex-4k"

    df, label_map = createDataCSV(dataset, fold=0, eval=True)
    num_samples, num_labels = df[df.dataType == "test"].shape[0], len(label_map)
    print(f'load {dataset} dataset with '
          f'{len(df[df.dataType == "train"])} train {len(df[df.dataType == "test"])} test with {len(label_map)} labels done')

    xmc_models = []
    prediction = np.zeros((num_samples, num_labels))
    # predictions = torch.zeros(num_samples, num_labels)
    berts = ['bert-base', 'roberta', 'xlnet']

    for index in range(len(berts)):
        model_name = [dataset, '' if berts[index] == 'bert-base' else berts[index]]
        model_name = '_'.join([i for i in model_name if i != ''])

        model = LightXML(n_labels=len(label_map), bert=berts[index])

        print(f'Loading {model_name}')
        model.load_state_dict(torch.load(f'./resource/model_checkpoint/model-{model_name}.bin'))

        tokenizer = model.get_tokenizer()
        test_d = MDataset(df, 'test', tokenizer, label_map,
                          128 if dataset == 'Amazoncat-13k' and berts[index] == 'xlnent' else 512)
        testloader = DataLoader(test_d, batch_size=16, num_workers=0,
                                shuffle=False)

        model.cuda()
        prediction = prediction + torch.sigmoid(torch.Tensor(
            model.one_epoch(0, testloader, None, mode='test')[0])).cpu().numpy()
        # xmc_models.append(model)

    np.save(f'./resource/prediction/LightXML_{dataset}_prediction.npy', prediction)

