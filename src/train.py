import torch
from torch import nn
import sys
from src import models
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
import glob
import pandas as pd
from RAdam.radam import RAdam

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.test == False:
        model = getattr(models, hyp_params.model+'Model')(hyp_params)
    else:
        load_model = './check_point/best_mult__MULT2.pt'
        model = torch.load(load_model)
        model.to(torch.float32)

    if hyp_params.use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    if hyp_params.optim == 'SGD':
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr, momentum= 0.5)
    elif hyp_params.optim == 'RAdam':
        optimizer = RAdam(model.parameters(), lr=hyp_params.lr)
    else:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)(reduction='sum')


    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):  
    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y) in enumerate(train_loader):
            idx, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)

            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk

            combined_loss = 0
            net = nn.DataParallel(model) if hyp_params.use_cuda else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)

                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                    
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                
                combined_loss = raw_loss
            else:
                preds, hiddens = net(text, audio, vision)
                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss
                combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item()
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
        return epoch_loss

    def evaluate(model, criterion, loader):
        model.eval()

        total_loss = 0.0
        results = []
        truths = []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y) in enumerate(loader):
                index_, text, audio, vision = batch_X
                eval_attr = batch_Y
                if hyp_params.use_cuda:
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                batch_size = text.size(0)

                # net = nn.DataParallel(model) if hyp_params.use_cuda else model
                net = model
                preds, output = net(text, audio, vision)
                loss = criterion(preds, eval_attr)
                total_loss += loss

                results.append(preds)
                truths.append(eval_attr)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return total_loss, results, truths
    
    def eval_metric(results, truths, exclude_zero=False):
        test_preds = results.reshape(-1).cpu().detach().numpy()
        test_truth = truths.reshape(-1).cpu().detach().numpy()
        test_preds = np.where(test_preds > 0.5, 1, 0)
        mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
        corr = np.corrcoef(test_truth, test_preds)[0][1]
        f_score = f1_score(test_truth, test_preds)
        acc = accuracy_score(test_truth, test_preds)
        recall = recall_score(test_truth, test_preds)
        precision = precision_score(test_truth, test_preds)

        print("   - F1 score : ", f_score, "   - recall   : ", recall, "   - prec     : ", precision, "   - ac score : ", acc)
        print("   - C-matrix : \n", confusion_matrix(test_preds, test_truth, labels=[1,0]))

    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    scheduler = settings['scheduler']

    best_valid = 0
    best_score = 1e8

    ###### Learning ###############
    
    if hyp_params.test == False:
        print("no")
        for epoch in range(1, hyp_params.num_epochs+1):
            start = time.time()
            train_loss = train(model, optimizer, criterion)
            print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f}'.format(epoch, time.time() - start, train_loss))
            end = time.time()
            duration = end-start
            
            val_loss, val_pred, val_truths = evaluate(model, valid_loader)
            scheduler.step(val_loss)    # Decay learning rate by validation loss 
            val_f_score, val_recall, val_precision, val_acc = eval_mosei_senti(val_pred, val_truths)
            
            test_loss, test_pred, test_truths = evaluate(model, test_loader)
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*25,"Evaluate Validation data", "-"*25)
        _, results, truths = evaluate(model, criterion, valid_loader)
        eval_metric(results, truths, True)

        print("-"*25,"Evaluate Test data", "-"*25)
        _, results, truths = evaluate(model, criterion, test_loader)
        results = results.cpu()
        eval_metric(results, truths, True)
        results = np.array(results) >= 0.5
        pd.DataFrame(results).to_csv(paths+"/test.bed", sep= '\t', index=False)
    else:
        _, results, truths = evaluate(model, criterion, test_loader)
        results = results.cpu()
        eval_metric(results, truths, True)
        results = np.array(results) >= 0.5
        pd.DataFrame(results).to_csv("./result.bed", sep= '\t', index=False)

    sys.stdout.flush()