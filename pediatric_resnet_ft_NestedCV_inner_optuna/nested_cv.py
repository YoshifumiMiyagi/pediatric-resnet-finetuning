from __future__ import annotations

import json, random
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .model import build_resnet50

class ImageTableDataset(Dataset):
    def __init__(self, df, image_col="path", label_col="label", transform=None):
        self.df=df.reset_index(drop=True); self.image_col=image_col; self.label_col=label_col; self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r=self.df.iloc[i]; x=Image.open(r[self.image_col]).convert("RGB")
        if self.transform: x=self.transform(x)
        return x, torch.tensor(int(r[self.label_col]),dtype=torch.long)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def default_transforms(size=224):
    n=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    return transforms.Compose([transforms.Resize((size,size)),transforms.RandomRotation(5),transforms.ToTensor(),n]), transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),n])

def _auc(y,p): return np.nan if len(np.unique(y))<2 else roc_auc_score(y,p)

def _epoch(model,loader,criterion,device,nout,optimizer=None):
    train=optimizer is not None; model.train(train); loss_sum=0.; ys=[]; ps=[]
    with (torch.enable_grad() if train else torch.no_grad()):
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            if train: optimizer.zero_grad(set_to_none=True)
            z=model(x)
            if nout==1: z=z.squeeze(1); loss=criterion(z,y.float()); p=torch.sigmoid(z)
            else: loss=criterion(z,y); p=torch.softmax(z,1)[:,1]
            if train: loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(y); ys.extend(y.detach().cpu().numpy()); ps.extend(p.detach().cpu().numpy())
    ys=np.asarray(ys); ps=np.asarray(ps)
    return loss_sum/len(loader.dataset),_auc(ys,ps),ys,ps

def _split(n,seed,grouped):
    return StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed) if grouped else StratifiedKFold(n_splits=n,shuffle=True,random_state=seed)

def _model(pretrained,path,mode,reset,device,lr,wd,epochs):
    m=build_resnet50(pretrained=pretrained,pretrained_path=path,mode=mode,num_outputs=None,reset_head=reset).to(device)
    nout=int(m.fc.out_features)
    c=nn.BCEWithLogitsLoss() if nout==1 else nn.CrossEntropyLoss()
    o=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=lr,weight_decay=wd)
    s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs)
    return m,nout,c,o,s

def run_nested_cv(csv_path:str,pretrained="adult",pretrained_path:Optional[str]=None,mode="layer4",
                  outer_splits=5,inner_splits=5,n_trials=20,batch_size=32,image_col="path",label_col="label",
                  id_col=None,group_col=None,output_dir="results_nested_optuna",num_workers=2,image_size=224,
                  reset_head=False,seed=42,device=None,lr_range=(1e-5,5e-4),weight_decay_range=(1e-6,1e-3),
                  epoch_range=(5,30),verbose=True):
    """Nested CV: Optuna uses inner folds only; each outer test fold is untouched until final evaluation."""
    try: import optuna
    except ImportError as e: raise ImportError("Install Optuna with: pip install optuna") from e
    df=pd.read_csv(csv_path).reset_index(drop=True); gcol=group_col or id_col
    for c in [image_col,label_col]+([gcol] if gcol else []):
        if c not in df.columns: raise ValueError(f"CSV is missing column: {c}")
    y=df[label_col].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0,1}): raise ValueError("label must be 0/1")
    grouped=gcol is not None; groups=df[gcol].to_numpy() if grouped else None
    out=Path(output_dir); weights=out/"weights"; studies=out/"optuna_studies"
    out.mkdir(parents=True,exist_ok=True); weights.mkdir(exist_ok=True); studies.mkdir(exist_ok=True)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu"); trtf,vatf=default_transforms(image_size)
    kw=dict(batch_size=batch_size,num_workers=num_workers,pin_memory=device.startswith("cuda"))
    oof=np.full(len(df),np.nan); fold_assign=np.full(len(df),-1); rows=[]; trials_all=[]; hist=[]
    outer=_split(outer_splits,seed,grouped); oit=outer.split(df,y,groups) if grouped else outer.split(df,y)
    if verbose: print(f"Nested CV outer={outer_splits} inner={inner_splits} trials={n_trials}; outer test is tuning-blind.")
    for ofold,(tri,tei) in enumerate(oit):
        otr=df.iloc[tri].reset_index(drop=True); ote=df.iloc[tei].reset_index(drop=True); oy=otr[label_col].astype(int).to_numpy()
        og=otr[gcol].to_numpy() if grouped else None; oseed=seed+ofold*10000
        if verbose: print(f"\nOuter {ofold+1}/{outer_splits}: train={len(otr)} test={len(ote)}")
        def objective(trial):
            lr=trial.suggest_float("lr",*lr_range,log=True); wd=trial.suggest_float("weight_decay",*weight_decay_range,log=True); epochs=trial.suggest_int("epochs",*epoch_range)
            icv=_split(inner_splits,oseed+trial.number+100,grouped); iit=icv.split(otr,oy,og) if grouped else icv.split(otr,oy); aucs=[]
            for ifold,(iti,ivi) in enumerate(iit):
                seed_everything(oseed+trial.number*100+ifold)
                dltr=DataLoader(ImageTableDataset(otr.iloc[iti],image_col,label_col,trtf),shuffle=True,**kw)
                dlva=DataLoader(ImageTableDataset(otr.iloc[ivi],image_col,label_col,vatf),shuffle=False,**kw)
                m,nout,c,o,s=_model(pretrained,pretrained_path,mode,reset_head,device,lr,wd,epochs)
                va=np.nan
                for ep in range(1,epochs+1):
                    tl,ta,_,_=_epoch(m,dltr,c,device,nout,o); vl,va,_,_=_epoch(m,dlva,c,device,nout); s.step()
                    hist.append(dict(stage="inner",outer_fold=ofold,trial=trial.number,inner_fold=ifold,epoch=ep,train_loss=tl,train_auc=ta,val_loss=vl,val_auc=va,lr=lr,weight_decay=wd,epochs=epochs))
                aucs.append(va); del m
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            good=[a for a in aucs if not np.isnan(a)]; return float(np.mean(good)) if good else -1.
        study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=oseed)); study.optimize(objective,n_trials=n_trials)
        tdf=study.trials_dataframe(); tdf.insert(0,"outer_fold",ofold); tdf.to_csv(studies/f"outer_fold{ofold}_trials.csv",index=False); trials_all.extend(tdf.to_dict("records"))
        bp=study.best_params; lr=float(bp["lr"]); wd=float(bp["weight_decay"]); epochs=int(bp["epochs"])
        with open(studies/f"outer_fold{ofold}_best_params.json","w") as f: json.dump({**bp,"inner_best_auc":float(study.best_value)},f,indent=2)
        seed_everything(oseed+9999)
        dltr=DataLoader(ImageTableDataset(otr,image_col,label_col,trtf),shuffle=True,**kw); dlte=DataLoader(ImageTableDataset(ote,image_col,label_col,vatf),shuffle=False,**kw)
        m,nout,c,o,s=_model(pretrained,pretrained_path,mode,reset_head,device,lr,wd,epochs)
        for ep in range(1,epochs+1):
            tl,ta,_,_=_epoch(m,dltr,c,device,nout,o); s.step()
            hist.append(dict(stage="outer_refit",outer_fold=ofold,trial=np.nan,inner_fold=np.nan,epoch=ep,train_loss=tl,train_auc=ta,val_loss=np.nan,val_auc=np.nan,lr=lr,weight_decay=wd,epochs=epochs))
        wp=weights/f"outer_fold{ofold}.pth"; torch.save(m.state_dict(),wp); test_loss,test_auc,_,prob=_epoch(m,dlte,c,device,nout)
        oof[tei]=prob; fold_assign[tei]=ofold
        row=dict(outer_fold=ofold,n_train_images=len(tri),n_test_images=len(tei),inner_best_auc=float(study.best_value),outer_test_auc=test_auc,outer_test_loss=test_loss,best_lr=lr,best_weight_decay=wd,best_epochs=epochs,best_weight=str(wp))
        if grouped: row.update(n_train_groups=otr[gcol].nunique(),n_test_groups=ote[gcol].nunique())
        rows.append(row)
        if verbose: print(f"  inner best={study.best_value:.4f}; epochs={epochs}; OUTER TEST AUC={test_auc:.4f}")
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if np.isnan(oof).any(): raise RuntimeError("Incomplete OOF predictions")
    cols=([id_col] if id_col else [])+([gcol] if gcol and gcol!=id_col else [])+[image_col,label_col]
    odf=df[cols].copy(); odf["outer_fold"]=fold_assign.astype(int); odf["oof_prob"]=oof
    rdf=pd.DataFrame(rows); hdf=pd.DataFrame(hist); tdf=pd.DataFrame(trials_all); bdf=rdf[["outer_fold","inner_best_auc","best_lr","best_weight_decay","best_epochs"]]
    odf.to_csv(out/"oof_predictions.csv",index=False); rdf.to_csv(out/"outer_fold_metrics.csv",index=False); bdf.to_csv(out/"best_params.csv",index=False); hdf.to_csv(out/"training_history.csv",index=False); tdf.to_csv(out/"optuna_trials.csv",index=False)
    overall=_auc(y,oof)
    if verbose: print(f"\nNested-CV OOF AUC={overall:.4f}\nSaved: {out}")
    return dict(oof=odf,outer_fold_metrics=rdf,best_params=bdf,optuna_trials=tdf,history=hdf,oof_auc=overall)
