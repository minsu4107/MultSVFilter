#!/usr/bin/env python

import sys
import pandas as pd
from pandas import DataFrame as df
import os, io
import argparse
import glob

def caller_df(path):
    def read_vcf_as_df(path):
        def find_not_json(string):
            find_list = ["IMPRECISE","SECONDARY",'\"PRECISE']
            for sol in find_list:
                index = string.find(sol)
                if index == -1:
                    continue
                string = string[:index+len(sol)+1] + ':\"1\"'+ string[index+len(sol)+1:]
            return string

        with open(path, 'r') as f:
            lines = [l for l in f if not l.startswith('##')]
        integrated_SV = pd.read_csv(io.StringIO(''.join(lines)),dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,'QUAL': str, 'FILTER': str, 'INFO': str},sep='\t').rename(columns={'#CHROM': 'CHROM'})
        FORMAT = integrated_SV.FORMAT[0].split(":")
        if path.find("NA") != -1:
            indi_feature = integrated_SV.pop('NA12878')
        elif path.find("HG") != -1:
            indi_feature = integrated_SV.pop('HG002')

        col_list = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
        for col in col_list:
            if col not in list(integrated_SV.columns):
                col_list.pop(col_list.index(col))
        tmp_feat = indi_feature.apply(lambda x: x.split(":"))
        tmp_feat = tmp_feat.to_list()

        indi_df = pd.DataFrame(tmp_feat)
        integrated_SV = integrated_SV[col_list].reset_index(drop=True)
        info_strings = '{"' + integrated_SV.pop('INFO').str.split(';').str.join('","').str.replace('=','":"').str.replace("\"\",", "") + '"}' 
        info_strings = info_strings.apply(lambda x: find_not_json(x))
        info_df = pd.json_normalize(info_strings.apply(eval),errors="ignore")
        return pd.concat([integrated_SV, info_df, indi_df], axis = 1)
    
    def RefSeq_chr_change(df):
        chr_change = pd.read_csv('hg38.chrnames', delimiter = ' ', header=None).to_numpy()
        chr_change = chr_change.transpose()
        df.CHROM = df.CHROM.replace(chr_change[0], chr_change[1])
        chr_change = pd.read_csv('hg19.chrnames', delimiter = ' ', header=None).to_numpy()
        chr_change = chr_change.transpose()
        df.CHROM = df.CHROM.replace(chr_change[0], chr_change[1])
        
        chroms=["1", "2", "3", "4", "5", "6", "7", "8", "9","10", "11", "12", "13", "14", "15", "16", "17","18", "19", "20", "21","22"]
        
        return df[df.CHROM.isin(chroms)].reset_index(drop=True)
    
    def apply_fill_null(x):
        if x.isna().SVLEN == True and x.isna().END == False:
            x.SVLEN = abs(int(x.END) - x.POS)
            return x
        elif x.isna().SVLEN == False and x.isna().END == True:
            x.END = abs(x.POS + x.SVLEN)
            return x
        return x
    
    caller_dict = read_vcf_as_df(path)
    caller_dict = RefSeq_chr_change(caller_dict)
    caller_dict = caller_dict.apply(lambda x: apply_fill_null(x), axis =1)
    return caller_dict

def labelling_bed(gt_df, target_bed):
    y_arr = list()
    for idx, target in target_bed.iterrows():
        result = pd.DataFrame()
        if target.SVTYPE == 'DUP':
            gt_ranged = gt_df[(gt_df.CHROM == target.CHROM) & (gt_df.SVTYPE == target.SVTYPE)]
            get_near_two = gt_ranged.iloc[(gt_ranged.POS - target.POS).abs().argsort()[:2]]
            for idx, near in get_near_two.iterrows():
                l = max(near.POS, target.POS)
                r = min(near.END, target.END)
                if r-l <0:
                    continue
                if r-l / (near.END - near.POS) >= 0.9 and r-l / (target.END - target.POS) >= 0.9:
                    result = near
        elif target.SVTYPE == 'DEL':
            gt_ranged = gt_df[(gt_df.CHROM == target.CHROM) & (gt_df.SVTYPE == target.SVTYPE)]
            result = gt_ranged[(abs(gt_ranged.POS - target.POS) < 158) & (abs(gt_ranged.END - target.END) < 158)]
        if result.empty:
            y_arr.append(False)
        else:
            y_arr.append(True)
    target_bed['Y'] = y_arr
    return target_bed

def exclude(svset,excludeFile):
    excludeRecords=[]
    f = open(excludeFile, 'r')
    for line in f:
        record = line.split()

        chrom = record[0]
        start=int(record[1])
        end=int(record[2])
        tmp = interval(chrom, start, end, 'DEL')
        excludeRecords.append(tmp)
    records=[]
    for interval1 in svset:
        tag=False
        for interval2 in excludeRecords:
            if interval1.is_overlap(interval2):
                tag=True
                break
        if tag== False:
            records.append(interval1)
    return records

def make_bedFromCsv(prmt_df,path):
    tmp_df = prmt_df
    tmp_df = tmp_df[(tmp_df.FILTER == 'PASS') | (tmp_df.FILTER == '.')]
    tmp_df = tmp_df[['CHROM','POS','END',"SVTYPE"]]
    tmp_df.END.fillna(".",inplace = True)
    tmp_df = tmp_df[(tmp_df.SVTYPE == 'DUP') |(tmp_df.SVTYPE == 'DEL')].reset_index(drop = True)
    tmp_df['END'] = pd.to_numeric(tmp_df['END'])
    tmp_df['END'] = tmp_df['END'].astype(int)
    tmp_df = tmp_df[abs(tmp_df.POS - tmp_df.END) <= 2000000]
    
    # df[df.SVTYPE == 'DEL'].to_csv(path.replace(".csv",".bed"), sep = '\t', header = None, index = False)
    return tmp_df.reset_index(drop=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vcf_path', dest='vcf_path', required=True, help='vcf file path')
    parser.add_argument('--label_file', dest='label_file', required=True, help='label_file')
    args = parser.parse_args()

    vcf_glob = glob.glob(args.vcf_path+"/*.vcf")
    print(vcf_glob)

    for vcf_file in vcf_glob:
        result = caller_df(vcf_file)    
        result.to_csv(vcf_file.replace(".vcf",".csv"), sep = '\t', index = False)
        target_bed = make_bedFromCsv(result, vcf_file)
        gt_df = pd.read_csv(args.label_file, delimiter = '\t')
        labeled = labelling_bed(gt_df, target_bed)
        labeled.to_csv(vcf_file.replace(".vcf",".bed"), sep = '\t', index = False)
    sys.stdout.flush()