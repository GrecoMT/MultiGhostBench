import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
import pickle
import random
from index import Indexer
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from lightning import Fabric
from tqdm import tqdm
import argparse
from text_embedding import TextEmbeddingModel
from utils.Turing_utils import load_Turing
from utils.Deepfake_utils import load_deepfake
from utils.OUTFOX_utils import load_OUTFOX
from utils.M4_utils import load_M4
from utils.OOD_utils import load_OOD
from dataset  import PassagesDataset

def infer(passages_dataloder,fabric,tokenizer,model):
    if fabric.global_rank == 0 :
        passages_dataloder=tqdm(passages_dataloder,total=len(passages_dataloder))
        allids, allembeddings,alllabels= [],[],[]
    model.model.eval()
    with torch.no_grad():
        for batch in passages_dataloder:
            text,label,write_model,write_model_set,ids= batch
            encoded_batch = tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        padding="max_length",
                        # padding=True,
                        truncation=True,
                    )
            encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
            # output = model(**encoded_batch).last_hidden_state
            # embeddings = pooling(output, encoded_batch)  
            # print(encoded_batch)
            embeddings = model(encoded_batch)
            # print(encoded_batch['input_ids'].shape)
            embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(1))
            label = fabric.all_gather(write_model).view(-1)
            ids = fabric.all_gather(ids).view(-1)
            if fabric.global_rank == 0 :
                allembeddings.append(embeddings.cpu())
                allids.extend(ids.cpu().tolist())
                alllabels.extend(label.cpu().tolist())
    if fabric.global_rank == 0 :
        allembeddings = torch.cat(allembeddings, dim=0)
        epsilon = 1e-6
        emb_dict,label_dict={},{}
        allembeddings= F.normalize(allembeddings,dim=-1)
        for i in range(len(allids)):
            emb_dict[allids[i]]=allembeddings[i]
            label_dict[allids[i]]=alllabels[i]
        allids,allembeddings,alllabels=[],[],[]
        for key in emb_dict:
            allids.append(key)
            allembeddings.append(emb_dict[key])
            alllabels.append(label_dict[key])
        allembeddings = torch.stack(allembeddings, dim=0)
        return allids, allembeddings.numpy(),alllabels
    else:
        return [],[],[]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def test(opt):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("device:", device)
    if opt.device_num>1:
        fabric = Fabric(accelerator=device,devices=opt.device_num,strategy='ddp')
    else:
        fabric = Fabric(accelerator=device,devices=opt.device_num)
    fabric.launch()
    model = TextEmbeddingModel(opt.model_name).cuda()
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    new_state_dict={}
    for key in state_dict.keys():
        if key.startswith('model.'):
            new_state_dict[key[6:]]=state_dict[key]
    model.load_state_dict(state_dict)
    tokenizer=model.tokenizer
    if opt.mode=='deepfake':
        database = load_deepfake(opt.database_path)[opt.database_name]
    elif opt.mode=='OUTFOX':
        database=load_OUTFOX(opt.database_path)[opt.database_name]
    elif opt.mode=='Turing':
        database=load_Turing(opt.database_path)[opt.database_name]
    elif opt.mode=='OOD':
        #database=load_OOD(opt.database_path, opt.unseen_author)[opt.database_name]
        database=load_OOD(resource=opt.resource, language=opt.language, unseen_author=opt.unseen_author)[opt.database_name]
    elif opt.mode=='M4':
        database=load_M4(opt.database_path)[opt.database_name]+load_M4(opt.database_path)[opt.database_name.replace('train','dev')]
        
    passage_dataset = PassagesDataset(database,mode=opt.mode,need_ids=True)
    print(len(passage_dataset))

    passages_dataloder = DataLoader(passage_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True)
    passages_dataloder=fabric.setup_dataloaders(passages_dataloder)
    model=fabric.setup(model)

    train_ids, train_embeddings,train_labels = infer(passages_dataloder,fabric,tokenizer,model)
    fabric.barrier()

    if fabric.global_rank == 0:
        index = Indexer(opt.embedding_dim)
        index.index_data(train_ids, train_embeddings)
        label_dict={}
        for i in range(len(train_ids)):
            label_dict[train_ids[i]]=train_labels[i]

        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        index.serialize(opt.save_path)
        #save label_dict using pickle
        with open(os.path.join(opt.save_path, 'label_dict.pkl'), 'wb') as f:
            pickle.dump(label_dict, f)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--embedding_dim', type=int, default=768)

    parser.add_argument('--language', type=str)
    parser.add_argument('--resource', type=str)

    parser.add_argument('--mode', type=str, default='deepfake', help="deepfake,MGT or MGTDetect_CoCo, or OOD")
    parser.add_argument('--unseen_author', type=str, default=None)
    
    #parser.add_argument("--database_path", type=str, default="/home/heyongxin/LLM_detect_data/Deepfake_dataset/cross_domains_cross_models")
    parser.add_argument("--database_path", type=str, default="/data/projects/punim0478/matteogreco/multilingual-OOD-AA/DeTeCtive/src/database/")
    
    parser.add_argument('--database_name', type=str, default='train', help="train,valid,test,test_ood")
    #parser.add_argument("--model_path", type=str, default="/home/heyongxin/detect-LLM-text/DAT/pth/unseen_model/model_best_gpt35.pth",\
    #                     help="Path to the embedding model checkpoint")
    
    parser.add_argument("--model_path", type=str, default="",\
                         help="Path to the embedding model checkpoint")
    #parser.add_argument('--model_name', type=str, default="princeton-nlp/unsup-simcse-roberta-base", help="Model name")
    parser.add_argument('--model_name', type=str, default="ZurichNLP/unsup-simcse-xlm-roberta-base", help="Model name")
    parser.add_argument("--save_path", type=str, default="database", help="Path to save the database")
    
    parser.add_argument('--seed', type=int, default=0)
    opt = parser.parse_args()
    opt.model_path = f"/data/projects/punim0478/matteogreco/multilingual-OOD-AA/DeTeCtive/src/runs/{opt.resource}-{opt.language}-OOD-unseen-{opt.unseen_author}_v0/model_best.pth"
    opt.save_path = f"{opt.save_path}/database-{opt.resource}-{opt.language}-unseen-{opt.unseen_author}"
    assert f"unseen-{opt.unseen_author}" in opt.model_path
    set_seed(opt.seed)
    test(opt)
