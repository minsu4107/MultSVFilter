import os
from PIL import Image
from PIL import ImageDraw
import pysam
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from tqdm import tqdm
import argparse
import time
from multiprocessing import Pool
import parmap
import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('--bam', dest='bam',type = str, required=True)
parser.add_argument('--sv_type', dest='sv_type',type = str, required=True)
parser.add_argument('--bed_file', dest='bed_file',type = str, required=True, help='SV bed_file path')
parser.add_argument('--fasta', dest='fasta',type = str, required=True, help='hg38 or hg19')
parser.add_argument('--t', dest='t',type = int, default = 1, help='multiprocessing core')
parser.add_argument('--output',type = str, default = './multi_preprocess/')
parser.add_argument('--img_size', dest='img_size', type = int,default = 255)
parser.add_argument('--label_path', dest='label_path', type =str, required=True)
args = parser.parse_args()

def estimateInsertSizes(alignments=1000000):
    print("==================Estimate Insertion Size==================")
    inserts = []
    count = 0
    bam = pysam.AlignmentFile(args.bam, "rb")
    for read in bam:
        if read.is_proper_pair and read.is_paired  and read.is_read1 and (not read.is_unmapped) and (not read.mate_is_unmapped) and (not read.is_duplicate) and (not read.is_secondary) and (not read.is_supplementary):
            if (read.reference_start < read.next_reference_start and (not read.is_reverse) and read.mate_is_reverse) or (read.reference_start > read.next_reference_start and read.is_reverse and (not read.mate_is_reverse)):
                count += 1
                if count <= alignments:
                    inserts.append(abs(read.tlen))
                else:
                    break
    bam.close()
    inserts = sorted(inserts)
    total_num = len(inserts)
    l = int(0.05 * total_num)
    r = int(0.95 * total_num)
    
    inserts = inserts[l:r]
    insert_mean, insert_std = int(np.mean(inserts)), int(np.std(inserts))
    print("Mean of the insert size is ", insert_mean, "Standard deviation of the insert size is ", insert_std)
    return insert_mean, insert_std

def getDelDRPList(bam_path, deletion, patch_size, mean_insert_size, sd_insert_size):
    l_extend, r_extend = patch_size//2, patch_size-patch_size//2
    left_list=[]
    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam.fetch(deletion[0], deletion[1]-l_extend-patch_size, deletion[1]+r_extend):
        if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start < read.next_reference_start:
            insert_size=abs(read.tlen)
            if (not read.is_reverse) and read.mate_is_reverse and (insert_size - mean_insert_size) > 3 * sd_insert_size:
                left_list.append(read.qname)
    bam.close()
    right_list=[]
    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam.fetch(deletion[0], deletion[2]-l_extend, deletion[2]+r_extend+patch_size):
        if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start > read.next_reference_start:
            insert_size=abs(read.tlen)
            if read.is_reverse and (not read.mate_is_reverse) and (insert_size - mean_insert_size) > 3 * sd_insert_size:
                right_list.append(read.qname)
    bam.close()
    drplist=list(set(left_list).intersection(set(right_list)))
    return drplist

def getTandemDupDRPList(bam_path, duplication, patch_size):
    l_extend, r_extend = patch_size//2, patch_size-patch_size//2
    left_list=[]
    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam.fetch(duplication[0], duplication[1]-l_extend, duplication[1]+r_extend+patch_size):
        if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start < read.next_reference_start:
            if read.is_reverse and (not read.mate_is_reverse) :
                left_list.append(read.qname)
    bam.close()
    right_list=[]
    bam = pysam.AlignmentFile(bam_path, "rb")
    for read in bam.fetch(duplication[0], duplication[2]-l_extend-patch_size, duplication[2]+r_extend):
        if read.is_paired and (not read.is_unmapped) and (not read.mate_is_unmapped) and read.reference_start > read.next_reference_start:
            if (not read.is_reverse) and read.mate_is_reverse:
                right_list.append(read.qname)
    bam.close()
    drplist=list(set(left_list).intersection(set(right_list)))
    return drplist

def is_left_soft_clipped_read(read): 
    if(read.cigartuples[0][0]==4):
        return True
    else:
        return False

def is_right_soft_clipped_read(read):
    if(read.cigartuples[-1][0]==4):
        return True
    else:
        return False

def draw_deletion(bam_path, record, pic_start, pic_end, flag_LR, drp_list):
    scale_pix = 1 
    pic_length = (pic_end - pic_start)
    im = Image.new("RGB", [pic_length * scale_pix, (pic_length // 2 - 1) * scale_pix], "black") 
    im_draw = ImageDraw.Draw(im)

    column_statistics_list = np.zeros((pic_length, 3))
    both_count_arr = np.zeros((pic_length,1))
    bam = pysam.AlignmentFile(bam_path, "rb")
    read_depth = np.zeros((pic_length, 1))
    for read in bam.fetch(record[0], pic_start, pic_end):
        if read.is_unmapped:
            continue
        read_lr = (read.reference_start + 1, read.reference_end)

        flag_drp = 0
        if read.qname in drp_list:
            if str(flag_LR) == '1':
                if read.reference_start < read.next_reference_start:
                    flag_drp=1
            else:
                if read.reference_start > read.next_reference_start:
                    flag_drp=1

        flag_sr = 0
        if str(flag_LR) == '1':
            if is_right_soft_clipped_read(read):
                flag_sr=1
        else:
            if is_left_soft_clipped_read(read):
                flag_sr=11

        read_pic_l = (read_lr[0] - pic_start) if read_lr[0] >= pic_start else 0
        read_pic_r = (read_lr[1] - pic_start) if read_lr[1] <= pic_end else pic_length - 1
        
        for i in range(read_pic_l, read_pic_r):
            read_depth[i] += 1
            column_statistics_list[i][0] += 1
            if flag_drp == 1 and flag_sr == 1:
                both_count_arr[i] += 1
            elif flag_drp == 1:
                column_statistics_list[i][1] += 1
            elif flag_sr == 1:
                column_statistics_list[i][2] += 1
    bam.close()
    
    for x in range(len(column_statistics_list)):
        y = 0
        rd_count = column_statistics_list[x][0]
        drp_count = column_statistics_list[x][1]
        sr_count = column_statistics_list[x][2]
        both_count = both_count_arr[x]
        
        # SR&RP
        if both_count != 0:
            base_rgb = tuple([255, 255, 255])
            im_draw.rectangle((x * scale_pix, y, x * scale_pix + scale_pix, both_count * scale_pix), fill=base_rgb)
        # split read
        if sr_count != 0:
            base_rgb = tuple([255, 0, 255])
            im_draw.rectangle(
                (x * scale_pix, both_count * scale_pix, x * scale_pix + scale_pix, (both_count + sr_count) * scale_pix),
                fill=base_rgb)
        # discordant read pair
        if drp_count != 0:
            base_rgb = tuple([255, 255, 0])
            im_draw.rectangle(
                (x * scale_pix, (both_count + sr_count) * scale_pix, x * scale_pix + scale_pix,
                 (drp_count + sr_count + both_count) * scale_pix), fill=base_rgb)

        # read depth
        if rd_count != 0:
            base_rgb = tuple([255, 0, 0])
            im_draw.rectangle(
                (x * scale_pix, (drp_count + sr_count + both_count) * scale_pix, x * scale_pix + scale_pix,
                 (rd_count) * scale_pix),
                fill=base_rgb)  

    return im

def draw_tandem_duplication(bam_path, record, pic_start, pic_end, flag_LR, drp_list):
    scale_pix = 1 
    pic_length = (pic_end - pic_start)

    column_statistics_list = np.zeros((pic_length, 3))
                         
    bam = pysam.AlignmentFile(bam_path, "rb")
    max_number_of_reads=10000
    count=0
    read_depth = np.zeros((pic_length, 1))

    for read in bam.fetch(record[0], pic_start, pic_end):
        if count > max_number_of_reads:
            break
        count=count+1

        if read.is_unmapped:
            continue
        read_lr = (read.reference_start + 1, read.reference_end)

        flag_drp = 0
        if read.qname in drp_list:
            if read.qname in drp_list:
                if str(flag_LR) == '1':
                    if read.reference_start < read.next_reference_start:
                        flag_drp=1
                else:
                    if read.reference_start > read.next_reference_start:
                        flag_drp=1
        flag_sr = 0
        if str(flag_LR) == '1':
            if is_left_soft_clipped_read(read):
                flag_sr=1
        else:
            if is_right_soft_clipped_read(read):
                flag_sr=1

        read_pic_l = (read_lr[0] - pic_start) if read_lr[0] >= pic_start else 0
        read_pic_r = (read_lr[1] - pic_start) if read_lr[1] <= pic_end else pic_length - 1

        for i in range(read_pic_l, read_pic_r):
            read_depth[i] += 1
            column_statistics_list[i][0] += 1
            if flag_drp == 1:
                column_statistics_list[i][1] += 1
            elif flag_sr == 1:
                column_statistics_list[i][2] += 1
    bam.close()

    return column_statistics_list, read_depth

# trans2img(args.bam, args.sv_type, args.bed, args.img_size, mean_size, std_size)
def trans_modality(bed_path):
    
    sv_list = pd.read_csv(bed_path, delimiter = '\t', dtype = {'CHROM':'str', 'POS':"int", 'END':"int", 'SVTYPE':"str"})
    sv_list = sv_list[sv_list['SVTYPE'] == args.sv_type]
    chroms = []
    if args.fasta.find('38') != -1:
        chroms = ['NC_000001.11', 'NC_000002.12', 'NC_000003.12', 'NC_000004.12', 'NC_000005.10', 'NC_000006.12', 'NC_000007.14', 'NC_000008.11', 'NC_000009.12', 'NC_000010.11', 'NC_000011.10', 'NC_000012.12', 'NC_000013.11', 'NC_000014.9', 'NC_000015.10', 'NC_000016.10', 'NC_000017.11', 'NC_000018.10', 'NC_000019.10', 'NC_000020.11', 'NC_000021.9', 'NC_000022.11']
    elif args.fasta.find('37') != -1 or args.fasta.find('19') != -1:
        chroms = list(map(str,range(1,23)))
    
    sv_list['CHROM'] = sv_list['CHROM'].replace(list(map(str,range(1, 23))), chroms)
    sv_list = sv_list.to_numpy()
    print("[*] Start generating " + args.sv_type + " images ===")

    return draw_pic(sv_list)


def draw_pic(sv_list):
    os.makedirs(args.output+args.sv_type+'/image/', exist_ok = True)
    os.makedirs(args.output+args.sv_type+'/signal/', exist_ok = True)
    os.makedirs(args.output+args.sv_type+'/text/', exist_ok=True)

    p = parmap.map(do_draw, sv_list, pm_pbar=True, pm_processes=args.t)
    return

def do_draw(target_sv):
    chroms = []

    if args.fasta.find('38') != -1:
        chroms = ['NC_000001.11', 'NC_000002.12', 'NC_000003.12', 'NC_000004.12', 'NC_000005.10', 'NC_000006.12', 'NC_000007.14', 'NC_000008.11', 'NC_000009.12', 'NC_000010.11', 'NC_000011.10', 'NC_000012.12', 'NC_000013.11', 'NC_000014.9', 'NC_000015.10', 'NC_000016.10', 'NC_000017.11', 'NC_000018.10', 'NC_000019.10', 'NC_000020.11', 'NC_000021.9', 'NC_000022.11']
    elif args.fasta.find('37') != -1 or args.fasta.find('19') != -1:
        chroms = list(map(str,range(1,23)))
    patch_size = args.img_size
    l_extend, r_extend = patch_size//2, patch_size-patch_size//2
    bam_file = args.bam
    arrs = []
    signal_arrs = []
    seqs = []
    if args.sv_type=="DEL":
        drp_list=getDelDRPList(bam_file, target_sv, patch_size, mean_size, std_size)
        for flag_LR in [1, 2]:
            bp_position = int(target_sv[flag_LR])
            pic_start, pic_end = bp_position - l_extend, bp_position + r_extend
            arr, signal = draw_deletion(bam_file, target_sv, pic_start, pic_end, flag_LR, drp_list)
            seq = make_text_modality(target_sv, pic_start, pic_end)
            arrs.append(arr)
            seqs.append(seq)
            signal_arrs.append(signal)
    if args.sv_type=="DUP":
        drp_list=getTandemDupDRPList(bam_file, target_sv, patch_size)
        for flag_LR in [1, 2]:
            bp_position = int(target_sv[flag_LR])
            pic_start, pic_end = bp_position - l_extend, bp_position + r_extend
            seq = make_text_modality(target_sv, pic_start, pic_end)
            arr, signal =draw_tandem_duplication(bam_file, target_sv, pic_start, pic_end, flag_LR, drp_list)
            arrs.append(arr)
            seqs.append(seq)
            signal_arrs.append(signal)

    seq_np_arr = np.concatenate([np.array(seqs[0]), np.array(seqs[1])], axis = 0)
    save_path = args.output+args.sv_type + '/text/'+ args.sv_type + '_' +'chr'+str(chroms.index(target_sv[0])+1) + '_' + str(target_sv[1]) + '_' + str(target_sv[2]) + ".npy"
    np.save(save_path,seq_np_arr)  

    image_np_arr = np.concatenate([np.array(arrs[0]), np.array(arrs[1])], axis = 0)
    save_path = args.output+args.sv_type + '/image/'+ args.sv_type + '_' +'chr'+str(chroms.index(target_sv[0])+1)  + '_' + str(target_sv[1]) + '_' + str(target_sv[2]) + ".npy"
    np.save(save_path,image_np_arr)

    signal_np_arr = np.concatenate([np.array(rd_arrs[0]), np.array(rd_arrs[1])], axis = 0)
    save_path = args.output+args.sv_type + '/signal/'+ args.sv_type + '_' +'chr'+str(chroms.index(target_sv[0])+1) + '_' + str(target_sv[1]) + '_' + str(target_sv[2]) + ".npy"
    np.save(save_path,rd_np_arr)
    return

def make_text_modality(bed_path):
    os.makedirs(args.output+args.sv_type+'/text/', exist_ok=True)

    if args.fasta.find('38') != -1:
        chroms = ['NC_000001.11', 'NC_000002.12', 'NC_000003.12', 'NC_000004.12', 'NC_000005.10', 'NC_000006.12', 'NC_000007.14', 'NC_000008.11', 'NC_000009.12', 'NC_000010.11', 'NC_000011.10', 'NC_000012.12', 'NC_000013.11', 'NC_000014.9', 'NC_000015.10', 'NC_000016.10', 'NC_000017.11', 'NC_000018.10', 'NC_000019.10', 'NC_000020.11', 'NC_000021.9', 'NC_000022.11']
    elif args.fasta.find('37') != -1 or args.fasta.find('19') != -1:
        chroms = list(map(str,range(1,23)))
    def seq2arr(bp):
        upper = 1.0 if ord(bp) < 96 else 0.5
        if bp.lower() == 'a':
            return np.array([upper, -upper, upper*0.2, upper*(-0.2)])
        elif bp.lower() == 't':
            return np.array([-upper, upper, upper*(-0.2), upper*0.2])
        elif bp.lower() == 'g':
            return np.array([upper*0.2, upper*(-0.2), upper, -upper])
        elif bp.lower() == 'c':
            return np.array([upper*(-0.2), upper*(0.2), -upper, upper])
        else:
            return np.array([0, 0, 0, 0])
    
    fasta = pysam.FastaFile(args.fasta)
    bed = pd.read_csv(bed_path, delimiter = '\t')
    bed = bed[bed['SVTYPE'] != 'BND']
    bed['CHROM'] = bed['CHROM'].replace(range(1,23), chroms)
    for idx ,value in bed.iterrows():
        chrom, start, end, svtype, y = value
        for breakpoint in [start, end]:
            seq_l = fasta.fetch(chrom, start-128, start+127)
            seq_r = fasta.fetch(chrom, end-128, end+127)
            result_l = np.array(list(map(seq2arr,seq_l)))
            result_r = np.array(list(map(seq2arr,seq_r)))
            result = np.concatenate([result_l, result_r], axis =0)
            np.save(args.output+args.sv_type+'/text/'+args.sv_type+'_chr'+str(chroms.index(chrom)+1)+'_'+str(start)+'_'+str(end)+".npy", result)

if __name__ == '__main__':
    start = time.time()
    bed_file = args.bed_file
    global mean_size, std_size
    mean_size, std_size = estimateInsertSizes()
    trans_modality(bed_file)
    seq_preprocess(bed_file)
    sys.stdout.flush()